# waves


To use the accurate data, please download it from CDIP website.

Save the data at thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/ 
How to train the ml algorithms.


1) Run `python test_cdip.py 1` to get the data from the CDIP website.
2) Run `python test_stuff.py 3` to get the parameters for the ml algorithms.
3) Use the best parameters used and put it in save_files.py. And, then run `python save_files.py 3` to save ml algorithms.
4) Run `python test.py filename` to run testing on the data. Note that the data should be in CDIP format for now.


Thanks,
Enjoy
