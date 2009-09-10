public class PAPI_multiplex_option {
  public EventSet eventset;
  public int us;
  public int max_degree;

  public PAPI_multiplex_option() {
    // do nothing
  }

  public PAPI_multiplex_option(EventSet set, int u, int m) {
    eventset=set;
    us=u;
    max_degree=m;
  }
}
