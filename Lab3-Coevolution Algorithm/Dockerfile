FROM pklehre/niso-lab3
ADD wxz739.py /bin
RUN apt-get update
RUN apt-get -y install python
RUN apt-get -y install python-numpy
RUN apt-get -y install python-scipy
CMD ["-username", "wxz739.py", "-submission", "python /bin/wxz739.py"]