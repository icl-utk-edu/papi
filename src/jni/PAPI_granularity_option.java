public class PAPI_granularity_option {
  public EventSet eventset;
  public int granularity;

  public PAPI_granularity_option() {
    // do nothing
  }

  public PAPI_granularity_option(EventSet set, int g) {
    eventset=set;
    granularity=g;
  }
}
