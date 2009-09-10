public class PAPI_mh_ucache_info extends PAPI_mh_cache_info{
  public PAPI_mh_ucache_info(int t, int s, int l, int n, int a){
    type=t;
    size=s;
    line_size=l;
    num_lines=n;
    associativity=a;
  }
  public void print_mh_ucache_info(){
    if(type==0) return;

    System.out.println("Unified Cache:");
    print_mh_cache_info();
  }
}
