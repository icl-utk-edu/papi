/*
 * pfmon_symbols.c  - management of symbol tables
 *
 * Copyright (C) 2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <elf.h>
/*
#include <libelf/libelf.h>
*/

#include "pfmon.h"

#ifndef ELF_ST_TYPE
#define ELF_ST_TYPE(val)         ((val) & 0xF)
#endif

typedef struct {
	char *name;
	unsigned long value;
	unsigned long size;
	int type;
} symbol_t;

static symbol_t *symbol_tab;
static	unsigned long number_of_symbols;
static  char *name_space;

static int
symcmp(const void *a, const void *b)
{
	symbol_t *ap = (symbol_t *)a;
	symbol_t *bp = (symbol_t *)b;

	return ap->value > bp->value;
}

static int
read_sym64(Elf *elf, unsigned long *nsyms)
{
	Elf_Data *data;
	Elf64_Shdr *symtab_hdr = NULL;
	Elf_Scn *symtab_section;
	Elf_Scn *strsym_section;
	Elf64_Sym *symtab_data;
	size_t strsym_index;
	int i, j, table_size;

    
	/* first find the symbol table table */
	symtab_section = NULL;
	while ((symtab_section = elf_nextscn(elf, symtab_section)) != 0) {

		symtab_hdr = elf64_getshdr(symtab_section);
		if (symtab_hdr == NULL) {
			vbprintf("cannot get section header\n");
			return -1;
		}
			
		/* is this the symbol table? no DYNSYMTAB? */
		if (symtab_hdr->sh_type == SHT_SYMTAB) goto found;
	}
	vbprintf("no symbol table found\n");
	return -1;
found:
	/* use elf_rawdata since there is no memory image of this data */
	data = elf_rawdata(symtab_section, NULL); 
	if (data == NULL) {
		vbprintf("can't extract raw elf data for symbol table\n");
		return -1;
	}
	symtab_data = (Elf64_Sym *)data->d_buf;
	table_size  = symtab_hdr->sh_size/symtab_hdr->sh_entsize;
  
	/* get the string table */
	strsym_index   = symtab_hdr->sh_link;
	strsym_section = elf_getscn(elf, strsym_index);
  
	/* use elf_rawdata since there is no memory image of this data */
	data = elf_rawdata(strsym_section, NULL); 
	if (data == NULL) {
		vbprintf("can't extract raw elf data for string section\n");
		return -1;
	}
  
	/* allocate space and copy content */
	name_space = malloc(data->d_size);
	if (name_space  == NULL) {
		vbprintf("can't allocate space for string table\n"); 
		return -1;
	}
  	memcpy(name_space, data->d_buf, data->d_size);
  
  	/* allocate space for the table and set it up */
	symbol_tab = malloc(table_size * sizeof(symbol_t));
	if (symbol_tab == NULL) {
    		vbprintf("cannot allocate space for symbol table\n");
		return -1;
	}
  	for (i = 0, j= 0; i < table_size; i++) {
    		symbol_tab[j].name  = name_space + symtab_data[i].st_name;
		if (symbol_tab[j].name == NULL || symbol_tab[j].name[0] == '\0') continue;
    		symbol_tab[j].value = symtab_data[i].st_value;
		symbol_tab[j].size  = symtab_data[i].st_size;
		symbol_tab[j].type  = symtab_data[i].st_info;
		j++;
	}
  	*nsyms = j;
	qsort(symbol_tab, j, sizeof(symbol_t), symcmp);
	return 0;
}


int
load_symbols(char *filename)
{
	Elf *elf;
	char *eident;
	int fd;
	static int symbols_loaded;

	if (symbols_loaded) return 0;

	fd = open(filename, O_RDONLY);
	if (fd == -1) fatal_error("symbol file %s not found\n", filename);

  	/* initial call to set internal version value */
	if (elf_version(EV_CURRENT) == EV_NONE)
		fatal_error("ELF library out of date");

  	/* prepare to read the entire file */
	elf = elf_begin(fd, ELF_C_READ, NULL);
	if (elf == NULL)
		fatal_error("can't read %s\n", filename);

	/* error checking */
	if (elf_kind(elf) != ELF_K_ELF)
		fatal_error("%s is not an ELF file\n", filename);
  
	eident = elf_getident(elf, NULL);
	if (eident[EI_MAG0] != ELFMAG0
	    || eident[EI_MAG1] != ELFMAG1
	    || eident[EI_MAG2] != ELFMAG2
	    || eident[EI_MAG3] != ELFMAG3)
		fatal_error("invalid ELF magic in %s\n", filename);

	switch (eident[EI_CLASS]) {
#if 0
  		case ELFCLASS32:
    			vbprintf("file class 32\n");
    			size = read_table32();
			break;
#endif
		case ELFCLASS64: 
    			//vbprintf("file class 64\n");
    			break;        
    		default:
    			fatal_error("unknown ELF class for %s\n", filename);
	}
	if (read_sym64(elf, &number_of_symbols)) fatal_error("cannot extract symbols from %s\n", filename);

	vbprintf("%s %ld symbols\n", filename, number_of_symbols);

	symbols_loaded = 1;

	return 0;
}

void
print_symbols(void)
{
	int i;

	for (i = 0; i < number_of_symbols; i++) {
		printf("0x%08lx %8lu %s\n", 
			symbol_tab[i].value,
			symbol_tab[i].size,
			symbol_tab[i].name);
	}
}

int
find_symbol_addr(char *name, int type, unsigned long *start, unsigned long *end)
{
	long i;

	if (name == NULL || *name == '\0' || start == NULL) return PFMLIB_ERR_INVAL;

	for (i = 0; i < number_of_symbols; i++) {
		if (!strcmp(name, symbol_tab[i].name) && ELF_ST_TYPE(symbol_tab[i].type) == type) 
			goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*start = symbol_tab[i].value;
	if (end) {
		if (symbol_tab[i].size != 0) {
			*end = *start +symbol_tab[i].size; 
			//vbprintf("symbol %s: [0x%lx-0x%lx)=%ld bytes\n", name, *start, *end, symbol_tab[i].size);
		} else {
			vbprintf("program compiled with buggy toolchain, using approximation for symbol %s\n", name);
			if (i == (number_of_symbols-1)) return PFMLIB_ERR_NOTFOUND;
		        /*
		 	 * XXX: Very approximative and maybe false at times
		 	 * Use carefully
		 	 */
			*end = symbol_tab[i+1].value;
		}
		vbprintf("symbol %s (%s): [0x%lx-0x%lx)=%ld bytes\n", 
				name, 
				type == STT_FUNC ? "code" : "data",
				*start, 
				*end, 
				*end-*start);
	}
	return PFMLIB_SUCCESS;
}

int
find_code_symbol_addr(char *name, unsigned long *start, unsigned long *end)
{
	return find_symbol_addr(name, STT_FUNC, start, end);
}

int
find_data_symbol_addr(char *name, unsigned long *start, unsigned long *end)
{
	return find_symbol_addr(name, STT_OBJECT, start, end);
}
