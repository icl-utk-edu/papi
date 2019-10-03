#ifndef _EVENT_STOCK_
#define _EVENT_STOCK_

typedef struct
{
    int size;
    int* evtsizes;
    unsigned int* maxqualsize;
    char** base_evts;
    char*** evts;
} evstock;

int          build_stock(evstock* stock, int cap);
void         print_stock(evstock* stock);
int          num_evts(evstock* stock);
int          num_quals(evstock* stock, int base_evt);
unsigned int max_qual_size(evstock* stock, int base_evt);
char*        evt_qual(evstock* stock, int base_evt, int tag);
char*        evt_name(evstock* stock, int index);
void         remove_stock(evstock* stock);

#endif
