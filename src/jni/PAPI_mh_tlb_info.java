public class PAPI_mh_tlb_info {
  public int type; /* Empty, instr, data, vector, unified */
  public int num_entries;  
  public int associativity;

  public PAPI_mh_tlb_info() {
    type=0;   
  }

  public PAPI_mh_tlb_info(int t, int n, int a){
    type=t;
    num_entries=n;
    associativity=a;
  }

  public void print_mh_tlb_info(){
    if(type==0) return;

    System.out.println("  Number of Entries: "+ num_entries);
    System.out.println("  Associativity : "+ associativity);
  }
}
