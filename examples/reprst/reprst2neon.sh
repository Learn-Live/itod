#!/bin/sh
echo "conduct experiments in a server named \"neon\""

### note: run xxx.sh out of the 'examples' directory, i.e. examples/reprst.sh
### keep 'the main function' still running after ending ssh session by execuate it in 'tumx'

### install third-part libraries if necessary
#pip3.7 install colorama --user
#pip3.7 install pyod --user # without needing root
#pip3.7 install jinja2 --user
#pip3.7 install xlrd --user
#pip3.7 install memory_profiler --user
#pip3.7 install h5py --user
#pip3.7 install pretty_errors --user
#pip3.7 install group-lasso --user
#pip3.7 install pympler --user
#pip3.7 install xlsxwriter --user
#pip3.7 install scikit-learn==0.22 --user
#pip3.7 install pyod==0.7.7.1 --user
#pip3.7 install tensorflow_gpu==2.0.0 --user
#pip3.7 install tensorboard --user
#pip3.7 install keras --user
#pip3.7 install torch==1.4.0 --user
#pip3.7 install torchvision==0.5.0 --user
#pip3.7 install colorlog==4.1.0 --user
#pip3.7 install colorlog==1.6.1 --user

# show current directory
echo 'PWD:' $PWD
### add root directory in PATH
PATH=$PATH:$PWD # add current directory to PATH
echo 'PATH:' $PATH

### run main function
exec_time=$(date +'%Y-%m-%d_%H:%M:%S')
echo $exec_time

##chmod 755 ./examples/mem_stats.sh
##./examples/mem_stats.sh > "mem_log_$(date +%Y-%m-%d_%H:%M:%S).txt" 2>&1 &
#./examples/mem_stats.sh > mem_log_${exec_time}.txt 2>&1 &

#for detector in GMM AE OCSVM KDE IF PCA;
for detector in AE; do
  if [ $detector = GMM ]; then # only GMM needs quickshift
#    ### install QuickshiftPP
#    cd itod/detector/postprocessing/quickshift/
#    echo $PWD
#    ## remove old build files first
#    ls ./
#    echo "**** rm -rf build/ *****"
#    rm -rf build/
#    rm -f quickshift_pp.cpp
#    ls ./
#    ## rebuild
#    python3.7 setup.py build
#    python3.7 setup.py install --user
#    cd ../../../../ # work root directory: IoT_feature_sets_comparison_20190822/

    # show current directory
    echo 'PWD:' $PWD
  fi

  # run under "examples"
#  cd examples/
  # src
   echo 'PYTHONPATH=../:./ python3.7 -u reprst/main_reprst_srcip.py -d' ${detector}' > ./out/reprst_srcip_new/'${detector}'_'${exec_time}'_log.txt 2>&1 &'
   PYTHONPATH=../:./ python3.7 -u reprst/main_reprst_srcip.py -d ${detector} >./out/reprst_srcip_new/${detector}_log.txt 2>&1 &
## both
#  echo 'PYTHONPATH=../:./ python3.7 -u reprst/main_reprst.py -d' ${detector}' > ./out/reprst/'${detector}'_'${exec_time}'_log.txt 2>&1 &'
#  PYTHONPATH=../:./ python3.7 -u reprst/main_reprst.py -d ${detector} >./out/reprst/${detector}_log.txt 2>&1 &
  # -u: write to file immediately
  # 2>&1: save stdout and stderr into a same file. I.e. > redirects stdout, 2>&1 redirects stderr to
  #       the same place as stdout
  # &: run in background

  #    status=$?   # only get the last process exitcode, and store exit status of grep to a variable 'status'
  #    echo 'exitcode:' $status

done
# End of script
