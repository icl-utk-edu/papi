public class PAPI_domain_option {
  public EventSet eventset;
  public int domain;

  public PAPI_domain_option() {
    // do nothing
  }

  public PAPI_domain_option(EventSet set, int d) {
    eventset=set;
    domain=d;
  }
}
