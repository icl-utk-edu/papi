public class PAPI_substrate_info {
  public int supports_program;        /* We can use programmable events */
  public int supports_write;          /* We can write the counters */
  public int supports_hw_overflow;    /* Needs overflow to be emulated */
  public int supports_hw_profile;     /* Needs profile to be emulated */
  public int supports_multiple_threads;     /* hardware counters support
                                            multiple threads */
  public int supports_64bit_counters; /* Only limited precision is available from hardware */
  public int supports_inheritance;    /* We can pass on and inherit child counters/values */
  public int supports_attach;         /* We can attach PAPI to another process */
  public int supports_real_usec;      /* We can use the real_usec call */
  public int supports_real_cyc;       /* We can use the real_cyc call */
  public int supports_virt_usec;      /* We can use the virt_usec call */
  public int supports_virt_cyc;       /* We can use the virt_cyc call */

  public PAPI_substrate_info() {
    // do nothing
  }

  public PAPI_substrate_info(int p, int w, int ho, int hp, int m, int c64, int i, int a, int ru, int rc, int vu, int vc) {
    supports_program=p;
    supports_write=w;
    supports_hw_overflow=ho;
    supports_hw_profile=hp;
    supports_multiple_threads=m;
    supports_64bit_counters=c64;
    supports_inheritance=i;
    supports_attach=a;
    supports_real_usec=ru;
    supports_real_cyc=rc; 
    supports_virt_usec=vu;
    supports_virt_cyc=vc;
  }
}
