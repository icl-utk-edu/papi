public class PAPI_mh_tlb_info {
  public int type; /* Empty, instr, data, vector, unified */
  public int num_entries;  
  public int associativity;

  public PAPI_mh_tlb_info() {
    // do nothing
  }

  public PAPI_mh_tlb_info(int t, int n, int a){
    type=t;
    num_entries=n;
    associativity=a;
  }
}
