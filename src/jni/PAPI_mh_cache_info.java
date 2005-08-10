public class PAPI_mh_cache_info {
  public int type; /* Empty, instr, data, vector, unified */
  public int size;  
  public int line_size;  
  public int num_lines;  
  public int associativity;

  public PAPI_mh_cache_info() {
    type=0;
  }

  public PAPI_mh_cache_info(int t, int s, int l, int n, int a){
    type=t;
    size=s;
    line_size=l;
    num_lines=n;
    associativity=a;
  }

  public void print_mh_cache_info(){
    if(type==0) return;

    System.out.println("  Size          : "+ size);
    System.out.println("  Line size     : "+ line_size);
    System.out.println("  Number of Line: "+ num_lines);
    System.out.println("  Associativity : "+ associativity);
  }
}
