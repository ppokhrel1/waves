# waves


To use the most recent data, please download it from CDIP website`http://cdip.ucsd.edu/`.

Save the data at thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/ . 
\n
How to train the ml algorithms.

1) Run `pip install -r requirements.txt`.
1) Run `python get_cdip_features.py 3` to get the data from the CDIP website.
2) Run `python test_classifiers.py 3` to get the parameters for the ml algorithms.
3) Use the best parameters used and put it in save_files.py. And, then run `python save_files.py 3` to save ml algorithms.
4) Run `python test.py filename` to run testing on the data. Note that the data should be in CDIP format for now.


Please cite:

`Pujan Pokhrel, Elias Ioup, Md Tamjidul Hoque, Mahdi Abdelguerfi, Julian Simeonov, "Forecasting significant wave heights in oceanic waters", 2021, SigKDD.
`

Thanks,
Enjoy
