#include "eventstock.h"

int build_stock(evstock* stock, int cap)
{
    int i,k;
    int ret;
    PAPI_event_info_t info;
    int cid;
    int ncomps = PAPI_num_components();
    int counter = 0;
    int subctr = 0;

    // Set the data stock's sizes all to zero.
    stock->size = 0;

    // If the cap is zero, then the user did not specify the number of events.
    // If CIT is using an input file, then to ensure each event in the file is
    // found, then the stock must contain all available events. If CIT is not
    // using an input file and the user did not specify the maximum number of
    // events to use, then a good default option is to let the stock contain
    // all available events.
    if(cap == 0)
    {
        cap = INT_MAX;
    }

    // Find the number of events.
    i=0 | PAPI_NATIVE_MASK;
    for(cid = 0; cid < ncomps && stock->size < cap; ++cid) // only use cap if cap > 0?
    {
        ret=PAPI_enum_cmp_event(&i,PAPI_ENUM_FIRST,cid);
        if(ret==PAPI_OK){
            do{
                memset(&info,0,sizeof(info));
                ret=PAPI_get_event_info(i,&info);
                if(ret != PAPI_OK)
                {
                    continue;
                }
                stock->size++;
            } while(PAPI_enum_cmp_event(&i,PAPI_ENUM_EVENTS,cid)==PAPI_OK && stock->size < cap);
        }
    }

    // Set the data stock's sizes all to zero.
    stock->evtsizes = (int*)calloc((stock->size),sizeof(int));
    stock->maxqualsize = (unsigned int*)calloc((stock->size),sizeof(unsigned int));
    stock->base_evts = (char**)malloc((stock->size)*sizeof(char*));

    // Stock names of the base events.
    i=0 | PAPI_NATIVE_MASK;
    for(cid = 0; cid < ncomps && counter < cap; ++cid)
    {
        ret=PAPI_enum_cmp_event(&i,PAPI_ENUM_FIRST,cid);
        if(ret==PAPI_OK){
            do{
                memset(&info,0,sizeof(info));
                ret=PAPI_get_event_info(i,&info);
                if(ret != PAPI_OK)
                {
                    continue;
                }
                stock->base_evts[counter] = strdup(info.symbol);
                counter++;
            } while(PAPI_enum_cmp_event(&i,PAPI_ENUM_EVENTS,cid)==PAPI_OK && counter < cap);
        }
    }

    // Find the number of qualifiers.
    i=0 | PAPI_NATIVE_MASK;
    counter = 0;
    for(cid = 0; cid < ncomps && counter < cap; ++cid)
    {
        // moving counter here fixes the issue but causes another.
        ret=PAPI_enum_cmp_event(&i,PAPI_ENUM_FIRST,cid);
        if(ret==PAPI_OK){
            do{
                memset(&info,0,sizeof(info));
                k=i;
                do{
                    ret=PAPI_get_event_info(k,&info);
                    if(ret != PAPI_OK)
                    {
                        continue;
                    }

                    if(strcmp(info.symbol, stock->base_evts[counter]) != 0 && strstr(info.symbol, "=") == NULL)
                    {    
                        stock->evtsizes[counter]++;
                    }

                }while(PAPI_enum_cmp_event(&k,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK);
                counter++;
            } while(PAPI_enum_cmp_event(&i,PAPI_ENUM_EVENTS,cid)==PAPI_OK && counter < cap);
        }
    }

    // Adjust the stock accordingly.
    stock->evts = (char***)malloc((stock->size)*sizeof(char**));
    for(i = 0; i < stock->size; ++i)
    {
        stock->evts[i] = (char**)malloc((stock->evtsizes[i])*sizeof(char*));
    }

    // Add the names to the stock.
    i=0 | PAPI_NATIVE_MASK;
    counter = 0;
    for(cid = 0; cid < ncomps && counter < cap; ++cid)
    {
        ret=PAPI_enum_cmp_event(&i,PAPI_ENUM_FIRST,cid);
        if(ret==PAPI_OK){
            do{
                memset(&info,0,sizeof(info));
                k=i;
                do
                {
                    ret=PAPI_get_event_info(k,&info);
                    if(ret != PAPI_OK)
                    {
                        continue;
                    }

                    if(strcmp(info.symbol, stock->base_evts[counter]) != 0 && strstr(info.symbol, "=") == NULL)
                    {    
                        char *col_pos = rindex(info.symbol, ':');
                        col_pos++;
                        stock->evts[counter][subctr] = strdup(col_pos);
                        subctr++;
                    }
                } while(PAPI_enum_cmp_event(&k,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK);
                subctr = 0;
                counter++;
            } while(PAPI_enum_cmp_event(&i,PAPI_ENUM_EVENTS,cid)==PAPI_OK && counter < cap);
        }
    }

    // Set max qualifier sizes.
    for(i = 0; i < stock->size; ++i)
    {
        for(k = 0; k < stock->evtsizes[i]; ++k)
        {
            if(strlen(stock->evts[i][k]) > stock->maxqualsize[i])
            {
                stock->maxqualsize[i] = strlen(stock->evts[i][k]);
            }
        }

        stock->maxqualsize[i]++;    // to capture null character termination
    }

    return 0;
}

void print_stock(evstock* stock)
{
    int i, j;
    for(i = 0; i < stock->size; ++i)
    {
        fprintf(stdout, "BASE EVENT <%s>\n", stock->base_evts[i]);
        for(j = 0; j < stock->evtsizes[i]; ++j)
        {
            fprintf(stdout, "%s\n", stock->evts[i][j]);
        }
    }

    return;
}

int num_evts(evstock* stock)
{
    return stock->size;
}

int num_quals(evstock* stock, int base_evt)
{
    return stock->evtsizes[base_evt];
}

unsigned int max_qual_size(evstock* stock, int base_evt)
{
    return stock->maxqualsize[base_evt];
}

char* evt_qual(evstock* stock, int base_evt, int tag)
{
    return stock->evts[base_evt][tag];
}

char* evt_name(evstock* stock, int index)
{
    return stock->base_evts[index];
}

void remove_stock(evstock* stock)
{
    int i, j;
    for(i = 0; i < stock->size; ++i)
    {
        for(j = 0; j < stock->evtsizes[i]; ++j)
        {
            free(stock->evts[i][j]);
        }

        free(stock->evts[i]);
        free(stock->base_evts[i]);
    }

    free(stock->evts);
    free(stock->base_evts);
    free(stock->evtsizes);
    free(stock->maxqualsize);
    free(stock);

    return;
}
