### A python library, "IoT Outlier Detection (itod)", is created for network novelty detection, which mainly includes two submodules: pcap parpser and novelty detection models. 
# Architecture:
    - docs/: 
        includes all documents (such as APIs)
    - examples/: 
        includes toy examples and datasets for you to play with it 
    - itod/: 
        source codes: includes two sublibraries (data and ndm)
        - ndm/: 
            includes different detection models (such as OCSVM)
        - data/: 
            includes pcap propcess (feature extraction from pcap) 
    - scripts/: 
        others (such as xxx.sh, make) 
    - tests/: 
        includes test cases
    - utils/: 
        includes common functions (such as load data and dump data)
    - visul/: 
        includes visualization functions
    - LICENSE.txt
    - readme.md
    - requirements.txt
    - setup.py
    - version.txt
   
<!--    
# How to install?
```
    pip3 install . 
    (pip3 will call setup.py to install the library automatically)
```
-->

# Experiment results 
```python3
    cd examples
    python3 -V
    PYTHONPATH=../:./ python3 reprst/main_reprst.py > out/main_reprst.txt 2>&1
```

- For more examples, please check the 'examples' directory 
    


# TODO
The current version just implements basic functions. We still need to further evaluate and optimize them continually. 
- Evaluate 'data' performance on different pcaps
- Add 'test' cases
- Add license
- Generated docs from docs-string automatically


Welcome to make any comments to make it more robust and easier to use!
