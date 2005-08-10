public class PAPI_mh_level_info {
  public PAPI_mh_itlb_info itlb;
  public PAPI_mh_dtlb_info dtlb;
  public PAPI_mh_utlb_info utlb;
  public PAPI_mh_icache_info icache;  
  public PAPI_mh_dcache_info dcache;  
  public PAPI_mh_ucache_info ucache;  

  public PAPI_mh_level_info() {
    // do nothing
  }

  public PAPI_mh_level_info(PAPI_mh_itlb_info it, PAPI_mh_dtlb_info dt, PAPI_mh_utlb_info ut, PAPI_mh_icache_info ic, PAPI_mh_dcache_info dc, PAPI_mh_ucache_info uc){
    itlb=it;
    icache=ic;
    dtlb=dt;
    dcache=dc;
    utlb=ut;
    ucache=uc;
  }

  public void print_mh_level_info(){
    itlb.print_mh_itlb_info();
    dtlb.print_mh_dtlb_info();
    utlb.print_mh_utlb_info();
    icache.print_mh_icache_info();
    dcache.print_mh_dcache_info();
    ucache.print_mh_ucache_info();
  }
}
