#!/usr/bin/python2
import urllib
import time
for i in range (1,10000000):
    print i;
    urllib.urlretrieve("https://webmail1.iitk.ac.in//squirrelmail/plugins/captcha/backends/phpcaptcha/image.php?sq="+str(i), str(i)+".jpg")
    time.sleep(0.5);
