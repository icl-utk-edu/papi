install_papi()
{
    ./configure --prefix=$PWD/test-install "$@"
    make && make install
}

test_papi_installation() 
{
    # Verify the installed component(s) are active


    # Run the installed component(s) tests
    case $components in
      amd_smi)
        bash components/amd_smi/tests/runtest.sh
        ;;
 
      cuda)
        bash components/cuda/tests/run_cuda_tests.sh 
        ;;
    
      rocp_sdk)
        bash components/rocp_sdk/tests/run_rocp_sdk_tests.sh
        ;;

     topdown)
       bash components/topdown/tests/run_topdown_tests.sh
       ;;
    
      *)
       echo "The provided component(s) ($components) is not a valid option. Exiting."
       exit 1
       ;; 
    esac
}
