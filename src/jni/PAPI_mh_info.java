public class PAPI_mh_info {
  public int levels;
  public PAPI_mh_level_info [] level;  

  public PAPI_mh_info() {
    // do nothing
  }

  public PAPI_mh_info(int l){
    levels=l;
    level=new PAPI_mh_level_info[levels];
  }

  public void mh_level_value(int l, PAPI_mh_level_info v){
    level[l] = v;
  }
}
