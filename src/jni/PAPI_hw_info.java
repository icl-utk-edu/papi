public class PAPI_hw_info {
  public int ncpu;             /* Number of CPU's in an SMP Node */
  public int nnodes;           /* Number of Nodes in the entire system */
  public int totalcpus;        /* Total number of CPU's in the entire system */
  public int vendor;           /* Vendor number of CPU */
  public String vendor_string; /* Vendor string of CPU */
  public int model;            /* Model number of CPU */
  public String model_string;  /* Model string of CPU */
  public float revision;       /* Revision of CPU */ 
  public float mhz;            /* Cycle time of this CPU, *may* be estimated at
                                  init time with a quick timing routine */
  public PAPI_mh_info mem_hierarchy;  /* PAPI memory heirarchy description */

  public PAPI_hw_info() {
    // do nothing
  }

/*  public PAPI_hw_info(int nc, int nn, int tot, int v, String vs, int m,
     String ms, float r, float z)
  {
    ncpu = nc;
    nnodes = nn;
    totalcpus = tot;
    vendor = v;
    vendor_string = vs;
    model = m;
    model_string = ms;
    revision = r;
    mhz = z;
  }
*/
  public PAPI_hw_info(int nc, int nn, int tot, int v, String vs, int m,
     String ms, float r, float z, PAPI_mh_info mh)
  {
    ncpu = nc;
    nnodes = nn;
    totalcpus = tot;
    vendor = v;
    vendor_string = vs;
    model = m;
    model_string = ms;
    revision = r;
    mhz = z;
    mem_hierarchy=mh;
  }
}
