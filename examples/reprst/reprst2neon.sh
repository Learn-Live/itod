#!/bin/sh

exec_time=$(date +'%Y-%m-%d_%H:%M:%S')
echo "Conduct experiments on a server (named \"neon\") at $exec_time"

### root path should be "Your_project/."
cd ../../
### show current directory
echo "PWD:$PWD ?"
### add current directory to PATH
PATH=$PATH:$PWD
echo "PATH: $PATH"

### add permission
# chmod 755 examples/reprst/reprst2neon.sh

### run each configuration
for detector in GMM AE OCSVM KDE IF PCA; do
  #for detector in KDE; do
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

  ### only get the results on 'src' traffic
  echo 'python3.7 -u examplesreprst/main_reprst_srcip.py -d' ${detector}' > ./out/reprst_srcip/'${detector}'_'${exec_time}'_log.txt 2>&1 &'
  #PYTHONPATH=../:./ python3.7 -u reprst/main_reprst_srcip.py -d ${detector} >./out/reprst_srcip/${detector}_log.txt 2>&1 &
  python3.7 -u examples/reprst/main_reprst_srcip.py -d ${detector} >./out/reprst_srcip/${detector}_log.txt 2>&1 &

  ### get the results on 'src+dst' traffic
  #  echo 'PYTHONPATH=../:./ python3.7 -u reprst/main_reprst.py -d' ${detector}' > ./out/reprst/'${detector}'_'${exec_time}'_log.txt 2>&1 &'
  #  PYTHONPATH=../:./ python3.7 -u reprst/main_reprst.py -d ${detector} >./out/reprst/${detector}_log.txt 2>&1 &

  # -u: write to file immediately
  # 2>&1: save stdout and stderr into a same file. I.e. > redirects stdout, 2>&1 redirects stderr to
  #       the same place as stdout
done # end of for

### get the paper results

### End of the script
