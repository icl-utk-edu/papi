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

  public void print_mh_info(){
    int i;

   System.out.println("Memory Hierarchy Levels: "+ levels);
   for( i = 0; i < levels; i++ ){
       System.out.println("Level "+ (i+1) + " memory info:");
       level[i].print_mh_level_info();
    }
  }
}
