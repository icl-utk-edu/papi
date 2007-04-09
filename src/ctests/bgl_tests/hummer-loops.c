#include "stdio.h"

int fmuladdv(int n ,double *a,double *b,double *c, double *t) {
  /* This function call generates N FMUL and N FADD operations
   * 3N FP LOADS and N FP STORES (when compiled with optimization)
   * t = a * b + c
   */
  int i;
  for(i=n;i;i--) {
    asm volatile("fmul %0,%1,%2" : "=f"(*t) : "f"(*(a++)), "f"(*(b++)));
    asm volatile("fadd %0,%1,%2" : "=f"(*t) : "0"(*t), "f"(*(c++)));
    t++;
  }
  return n;
}

int fmaddvSimple(int n ,double *a,double *b,double *c, double *t) {
  /* This function call generates N FMADD operations
   * 3N FP LOADS and N FP STORES (when compiled with optimization)
   * t = a * b + c
   */
  int i;
  for(i=n;i;i--) {
    asm volatile("fmadd %0,%1,%2,%3" :
               "=f"(*(t++)) : "f"(*(a++)), "f"(*(b++)), "f"(*(c++)));
  }
  return n;
}

int fmaddv(int n ,double *a_in,double *b_in,double *c_in, double *t_in) {
  /* This function call generates N FMADD operations
   * 3N FP LOADS and N FP STORES
   * t = a * b + c
   */
  int i;
  register int dblsz;
  register double *pa, *pb, *pc, *pt;
  register double r0,r1,r2,r3;

  asm volatile("li %0, 8" : "=r"(dblsz));
  pa=a_in;
  pb=b_in;
  pc=c_in;
  pt=t_in;
  asm volatile("fmadd  %0, %1, %2, %3" : "=f"(*pt) : "f"(*pa), "f"(*pb), "f"(*pc));
  for(i=n-1;i>0;i--) {
    asm volatile("lfdux  %1, %0, %3" : "=r"(pa), "=f"(r0) : "0"(pa), "r"(dblsz));
    asm volatile("lfdux  %1, %0, %3" : "=r"(pb), "=f"(r1) : "0"(pb), "r"(dblsz));
    asm volatile("lfdux  %1, %0, %3" : "=r"(pc), "=f"(r2) : "0"(pc), "r"(dblsz));
    asm volatile("fmadd  %0, %1, %2, %3" : "=f"(r3) : "f"(r0), "f"(r1), "f"(r2));
    asm volatile("stfdux %1, %0, %3" : "=r"(pt) : "f"(r3), "0"(pt), "r"(dblsz));
  }
  return n;
}

int fpmaddv(int n ,double *a_in,double *b_in,double *c_in, double *t_in) {
  /* This function call generates N/2 FPMADD operations
   * t = a * b + c
   */
  int i;
  register int dblsz,quadsz;
  register double *pa, *pb, *pc, *pt;
  register double r0,r1,r2,r3;

  if(n%2){
    fprintf(stderr,"Not an even number of operands!\n");
    return -1;
  }

  if( ((unsigned) a_in | (unsigned) b_in | (unsigned) c_in | (unsigned) t_in ) & 0xFU) {
    fprintf(stderr,"Check your data alignment!\n");
    return -1;
  }

  asm volatile("li %0, 16" : "=r"(quadsz));

  pa=a_in;
  pb=b_in;
  pc=c_in;
  pt=t_in;

  asm volatile("lfpdx  %0, 0, %1" : "=f"(r0) : "r"(pa));
  asm volatile("lfpdx  %0, 0, %1" : "=f"(r1) : "r"(pb));
  asm volatile("lfpdx  %0, 0, %1" : "=f"(r2) : "r"(pc));
  asm volatile("fpmadd  %0, %1, %2, %3" : "=f"(r3) : "f"(r0), "f"(r1), "f"(r2));
  asm volatile("stfpdx %0, 0, %1" : : "f"(r3), "r"(pt));
  for(i=n/2-1;i;--i) {
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pa), "=f"(r0) : "0"(pa), "r"(quadsz));
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pb), "=f"(r1) : "0"(pb), "r"(quadsz));
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pc), "=f"(r2) : "0"(pc), "r"(quadsz));
    asm volatile("fpmadd  %0, %1, %2, %3" : "=f"(r3) : "f"(r0), "f"(r1), "f"(r2));
    asm volatile("stfpdux %1, %0, %3" : "=r"(pt) : "f"(r3), "0"(pt), "r"(quadsz));
  }
  return n;
}

int fpmuladdv(int n ,double *a_in,double *b_in,double *c_in, double *t_in) {
  /* This function call generates N/2 FPMUL and N/2 FPADD operations
   * t = a * b + c
   */
  int i;
  register int dblsz,quadsz;
  register double *pa, *pb, *pc, *pt;
  register double r0,r1,r2,r3;

  if(n%2){
    fprintf(stderr,"Not an even number of operands!\n");
    return -1;
  }

  if( ((unsigned) a_in | (unsigned) b_in | (unsigned) c_in | (unsigned) t_in ) & 0xFU) {
    fprintf(stderr,"Check your data alignment!\n");
    return -1;
  }

  asm volatile("li %0, 16" : "=r"(quadsz));

  pa=a_in;
  pb=b_in;
  pc=c_in;
  pt=t_in;

  asm volatile("lfpdx  %0, 0, %1" : "=f"(r0) : "r"(pa));
  asm volatile("lfpdx  %0, 0, %1" : "=f"(r1) : "r"(pb));
  asm volatile("fpmul  %0, %1, %2" : "=f"(r3) : "f"(r0), "f"(r1));
  asm volatile("stfpdx %0, 0, %1" : : "f"(r3), "r"(pt));
  for(i=n/2-1;i;--i) {
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pa), "=f"(r0) : "0"(pa), "r"(quadsz));
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pb), "=f"(r1) : "0"(pb), "r"(quadsz));
    asm volatile("fpmul  %0, %1, %2" : "=f"(r3) : "f"(r0), "f"(r1));
    asm volatile("stfpdux %1, %0, %3" : "=r"(pt) : "f"(r3), "0"(pt), "r"(quadsz));
  }

  pt=t_in;
  pc=c_in;
  asm volatile("lfpdx  %0, 0, %1" : "=f"(r0) : "r"(pt));
  asm volatile("lfpdx  %0, 0, %1" : "=f"(r1) : "r"(pc));
  asm volatile("fpadd  %0, %1, %2" : "=f"(r3) : "f"(r0), "f"(r1));
  asm volatile("stfpdx %0, 0, %1" : : "f"(r3), "r"(pt));
  for(i=n/2-1;i;--i) {
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pt), "=f"(r0) : "0"(pt), "r"(quadsz));
    asm volatile("lfpdux  %1, %0, %3" : "=r"(pc), "=f"(r1) : "0"(pc), "r"(quadsz));
    asm volatile("fpadd  %0, %1, %2" : "=f"(r3) : "f"(r0), "f"(r1));
    asm volatile("stfpdx %0, 0, %1" : : "f"(r3), "r"(pt));
  }

  return n;
}
