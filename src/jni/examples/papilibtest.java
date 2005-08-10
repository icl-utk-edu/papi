public class papilibtest {
  public static void main(String [] args) {
    PapiJ p = new PapiJ();
    PAPI_exe_info exeInfo = null;
    PAPI_hw_info hwInfo = null;
	System.out.println(PapiJ.PAPI_VER_CURRENT);

    int ret =  p.library_init(PapiJ.PAPI_VER_CURRENT);

    if(ret < 0) {
      System.out.println("library_init returns " + ret);
      System.exit(-1);
    }
   

    exeInfo = p.get_executable_info();

    System.out.println("Executable info...");
    System.out.println("Full name = " + exeInfo.fullname);
    System.out.println("Name = " + exeInfo.address_info.name);
    System.out.println("Text start = " + exeInfo.address_info.text_start);
    System.out.println("Text end = " + exeInfo.address_info.text_end);
    System.out.println("Data start = " + exeInfo.address_info.data_start);
    System.out.println("Data end = " + exeInfo.address_info.data_end);
    System.out.println("Bss start = " + exeInfo.address_info.bss_start);
    System.out.println("Bss end = " + exeInfo.address_info.bss_end);

    //System.out.println("and finally the preload env...");
    //System.out.println(exeInfo.lib_preload_env);


    hwInfo = p.get_hardware_info();
    System.out.println("Hardware info...");
    System.out.println("Num cpu = " + hwInfo.ncpu);
    System.out.println("Num nodes = " + hwInfo.nnodes);
    System.out.println("Total cpus = " + hwInfo.totalcpus);
    System.out.println("Vendor = " + hwInfo.vendor);
    System.out.println("Vendor string = " + hwInfo.vendor_string);
    System.out.println("Model = " + hwInfo.model);
    System.out.println("Model string = " + hwInfo.model_string);
    System.out.println("Revision " + hwInfo.revision);
    System.out.println("Mhz = " + hwInfo.mhz);
    hwInfo.mem_hierarchy.print_mh_info();
  }
}
