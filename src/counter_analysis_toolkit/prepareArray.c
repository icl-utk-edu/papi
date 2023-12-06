#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <unistd.h>
#include <assert.h>

#include "prepareArray.h"

volatile uintptr_t opt_killer_zero;
static void _prepareArray_sections_random(uintptr_t *array, long long len, long long stride, long long secSize);
static void _prepareArray_sequential(uintptr_t *array, long long len, long long stride);


/*
 * "stride" is in "uintptr_t" elements, NOT in bytes
 * Note: It is wise to provide an "array" that is aligned to the cache line size.
 */
int prepareArray(uintptr_t *array, long long len, long long stride, long long secSize, int pattern){
    assert( array != NULL );
    opt_killer_zero = (uintptr_t)( (len+37)/(len+36) - 1 );

    switch(pattern){
        case SECRND:
            _prepareArray_sections_random(array, len, stride, secSize);
            break;
        case SEQUEN:
            _prepareArray_sequential(array, len, stride);
            break;
        default:
            fprintf(stderr,"prepareArray() unknown array access pattern: %d\n",pattern);
            return -1;
            break;
    }
    return 0;
}

/*
 * "stride" is in "uintptr_t" elements, NOT in bytes
 * Note: It is wise to provide an "array" that is aligned to the cache line size.
 */
static void _prepareArray_sections_random(uintptr_t *array, long long len, long long stride, long long secSize){

    assert( array != NULL );

    long long elemCnt, maxElemCnt, sec, i;
    long long currElemCnt, uniqIndex, taken;
    uintptr_t **p, *next;
    long long currSecSize = secSize;
    long long secCnt = 1+len/secSize;
    long long *availableNumbers;

    p = (uintptr_t **)&array[0];

    maxElemCnt = currSecSize/stride;
    availableNumbers = (long long *)calloc(maxElemCnt, sizeof(long long));

    // For every section of the array
    for(sec=0; sec<secCnt; ++sec){
        // if we are at the last section, trim the size"
        if( sec == secCnt-1 )
            currSecSize = (len%secSize);

        for(i=0; i<maxElemCnt; i++)
            availableNumbers[i] = i;
 
        currElemCnt = currSecSize/stride;

        taken = 0;
        if( 0==sec ) // For the first section we have already picked "0", so we must pick one less element.
            taken = 1;
        long long remainingElemCnt = currElemCnt;

        for(elemCnt=0; elemCnt<currElemCnt-taken; ++elemCnt){
            long long index = taken + random() % (remainingElemCnt-taken); // skip the first "taken" elements
            // For the first section we skip zero as a choice (we already selected that before the loop.)
            uniqIndex = sec*secSize + stride*availableNumbers[index];
            // replace the chosen number with the last element.
            availableNumbers[index] = availableNumbers[remainingElemCnt-1];
            // shrink the effective array size so the last element "drops off".
            remainingElemCnt--;

            // conneect the link
            next = &array[uniqIndex];
            *p = next;
            p = (uintptr_t **)next;
        }
    }

    // close the circle by pointing the last element to the start
    next = &array[0];
    *p = next;

    free(availableNumbers);
    
    return;
}

/*
 * "stride" is in "uintptr_t" elements, NOT in bytes
 * Note: It is wise to provide an "array" that is aligned to the cache line size.
 */

static void _prepareArray_sequential(uintptr_t *array, long long len, long long stride){
    long long curr;
    uintptr_t **p, *next;

    p = (uintptr_t **)&array[0];

    // As many times as there are elements that should be filled up in the array 
    for(curr=0; curr<len; curr+=stride){
        next = &array[curr];
        *p = next;
        p = (uintptr_t **)next;
    }
    // close the circle by pointing the last element to the start
    next = &array[0];
    *p = next;
    
    return;
}

