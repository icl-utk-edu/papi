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
#include <ctype.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <libelf/libelf.h>

#include "pfmon.h"

#ifndef ELF_ST_TYPE
#define ELF_ST_TYPE(val)         ((val) & 0xF)
#endif

#define TEXT_SYMBOL	1
#define DATA_SYMBOL	2

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
		symbol_tab[j].type  = ELF_ST_TYPE(symtab_data[i].st_info) == STT_FUNC ? TEXT_SYMBOL : DATA_SYMBOL;
		j++;
	}
  	*nsyms = j;
	qsort(symbol_tab, j, sizeof(symbol_t), symcmp);
	return 0;
}


static void
load_elf_symbols(void)
{
	Elf *elf;
	char *eident;
	char *filename = options.symbol_file;
	int fd;

	fd = open(filename, O_RDONLY);
	if (fd == -1) fatal_error("symbol file %s not found\n", options.symbol_file);

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
    			fatal_error("unsupported ELF class for %s\n", filename);
	}
	if (read_sym64(elf, &number_of_symbols)) fatal_error("cannot extract symbols from %s\n", filename);

	vbprintf("loaded %lu symbols from ELF file %s\n", number_of_symbols, filename);
}

static char *
place_str(int length)
{
	static char *current_free, *current_end;
	char *tmp;
#define STR_CHUNK_SIZE	1024

	if (length >= STR_CHUNK_SIZE)
		fatal_error("sysmap load string is too long\n");

	/*
	 * XXX: that's bad, we do not keep track of previously allocated
	 * chunks, so we cannot free!
	 */
	if (current_free == NULL || current_free-current_end < length) {
		current_free = (char *)malloc(STR_CHUNK_SIZE);
		if (current_free == NULL) return NULL;
		current_end = current_free + STR_CHUNK_SIZE;
	}
	tmp = current_free;
	current_free += length;
	return tmp;
}

/*
 * load a symbol file based on the System.map format (used by the Linux kernel)
 */
static void
load_sysmap_symbols(void)
{
	int fd;
	unsigned long nsyms = 0, idx = 0;
	unsigned long min_addr = 0UL;
	unsigned long line = 1UL;
	char *filename = options.symbol_file;
	char *p, *s, *end, *str_addr, *base;
	char *endptr;
	char b[24]; /* cannot be more than 16+2 (for 0x) */
	char type;
	int need_sorting = 0;
	struct stat st;


	fd = open(filename, O_RDONLY);
	if (fd == -1) fatal_error("sysmap  file %s not found\n", filename);

	if (fstat(fd, &st) == -1) fatal_error("cannot access sysmap file %s\n", filename);

	p = base = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (p == (char *)-1) fatal_error("cannot map sysmap file %s\n", filename);


	end = base + st.st_size;


	/* find number of symbols */
	while (p < end) {
		if (*p == '\n') nsyms++;
		p++;
	}
	symbol_tab = (symbol_t *)malloc(nsyms * sizeof(symbol_t));
	if (symbol_tab == NULL) fatal_error("cannot allocate sysmap table for %lu symbols\n", nsyms);

	idx = 0;

	/* now parse symbols */
	p = base;

	while (p < end) {

		/* find end */
		s = p;
		while(s < end && *s != ' ') s++;
		if (s == end) break;
		if (s-p > 16+2) fatal_error("invalid address at line %lu in %s\n", line, filename);

		strncpy(b, p, s-p);
		b[s-p] = '\0';

		/* point to object type */
		s++;
		type = tolower(*s);

		/* 
		 * keep only text and data symbols
		 * XXX: oversimplification here!
		 */
		if (type != 't' && type != 'd') {
			while(s < end && *s != '\n') s++;
			if (s == end) goto error;
			line++;
			p = s + 1;
			continue;
		}

		/* look for space separator */
		s++;
		if (*s != ' ') goto error;

    		symbol_tab[idx].type  = type == 't' ? TEXT_SYMBOL : DATA_SYMBOL;

		/* compute address */
    		symbol_tab[idx].value  = strtoul(b, &endptr, 16);
		if (*endptr != '\0') fatal_error("invalid address at line %lu in %s\n", line, filename);

		/*
		 * check that file is sorted correctly
		 */
		if (idx == 0) 
			min_addr = symbol_tab[idx].value;
		else if (symbol_tab[idx].value < min_addr) 
			need_sorting = 1;


		/* advance to symbol name */
		s++;
		p = s;	

		/* look for end-of-line */
		while(s < end && *s != '\n') s++;
		if (s == end) goto error;
		if (s == p) goto error;
		line++;

		/*
		 * place string in our memory pool
		 */
		str_addr = place_str(s-p+1);
		if (str_addr == NULL) goto error2;

		strncpy(str_addr, p, s-p);
		str_addr[s-p] = '\0';
		p = s +1;	

		/* sanity */
		if (idx == nsyms) fatal_error("too many symbol for sysmap files\n");

    		symbol_tab[idx].name  = str_addr;
    		symbol_tab[idx].size  = 0; /* use approximation */

		idx++;
	}
	/* record final number of symbols */
	number_of_symbols = idx;

	/*
	 * cleanup mappings
	 */
	munmap(base, st.st_size);
	close(fd);

	vbprintf("loaded %lu symbols from system.map file %s\n",  number_of_symbols, filename);

	/*
	 * normally a System.map file is already sort
	 * so we should not have to do this
	 */
	if (need_sorting) qsort(symbol_tab, idx, sizeof(symbol_t), symcmp);

	return;
error:
	fatal_error("sysmap file %s has invalid format, line %lu\n", filename, line);
error2:
	fatal_error("sysmap load file cannot place new string\n");
}

void
load_symbols()
{
	static int symbols_loaded;

	if (symbols_loaded) return;

	if (options.opt_sysmap_syms)
		load_sysmap_symbols();
	else
		load_elf_symbols();

	symbols_loaded = 1;
}

void
print_symbols(void)
{
	int i;

	for (i = 0; i < number_of_symbols; i++) {
		printf("0x%08lx %c %8lu %s\n", 
			symbol_tab[i].value, symbol_tab[i].type == TEXT_SYMBOL ? 'T' : 'D',
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
		if (!strcmp(name, symbol_tab[i].name) && symbol_tab[i].type == type) 
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
			/*
			 * with System.map we have no choice but to rely upon approximation
			 */
			if (options.opt_sysmap_syms == 0) 
				vbprintf("program compiled with buggy toolchain,");

			vbprintf("using approximation for size of symbol %s\n", name);

			if (i == (number_of_symbols-1)) {
				warning("cannot find another symbol to approximate size of %s\n", name);
				return PFMLIB_ERR_NOTFOUND;
			}

		        /*
		 	 * XXX: Very approximative and maybe false at times
		 	 * Use carefully
		 	 */
			*end = symbol_tab[i+1].value;
		}
		vbprintf("symbol %s (%s): [0x%lx-0x%lx)=%ld bytes\n", 
				name, 
				type == TEXT_SYMBOL ? "code" : "data",
				*start, 
				*end, 
				*end-*start);
	}
	return PFMLIB_SUCCESS;
}

int
find_code_symbol_addr(char *name, unsigned long *start, unsigned long *end)
{
	return find_symbol_addr(name, TEXT_SYMBOL, start, end);
}

int
find_data_symbol_addr(char *name, unsigned long *start, unsigned long *end)
{
	return find_symbol_addr(name, DATA_SYMBOL, start, end);
}
