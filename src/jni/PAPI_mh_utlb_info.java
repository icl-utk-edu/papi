public class PAPI_mh_utlb_info extends PAPI_mh_tlb_info{
  public PAPI_mh_utlb_info(int t, int n, int a){
    type=t;
    num_entries=n;
    associativity=a;
  }
  public void print_mh_utlb_info(){
    if(type==0) return;

    System.out.println("Unified TLB:");
    print_mh_tlb_info();
  }
}
