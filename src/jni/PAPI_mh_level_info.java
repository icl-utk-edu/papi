public class PAPI_mh_level_info {
  public PAPI_mh_itlb_info itlb;
  public PAPI_mh_dtlb_info dtlb;
  public PAPI_mh_icache_info icache;  
  public PAPI_mh_dcache_info dcache;  

  public PAPI_mh_level_info() {
    // do nothing
  }

  public PAPI_mh_level_info(PAPI_mh_itlb_info it, PAPI_mh_dtlb_info dt, PAPI_mh_icache_info ic, PAPI_mh_dcache_info dc){
    itlb=it;
    icache=ic;
    dtlb=dt;
    dcache=dc;
  }
}
