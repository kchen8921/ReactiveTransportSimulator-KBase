FROM kbase/sdkbase2:python
MAINTAINER KBase Developer
# -----------------------------------------
# In this section, you can install any system dependencies required
# to run your App.  For instance, you could place an apt-get update or
# install line here, a git checkout to download code, or run any other
# installation scripts.

# Install some libraries
RUN apt-get -y update && \
    apt-get install -y gfortran && \
    apt install -y mpich && \
    apt-get install -y libblas-dev liblapack-dev && \
    apt-get install -y libhdf5-dev
#RUN apt-get update
#RUN apt-get install -y gfortran
#RUN apt install -y mpich
#RUN apt-get install -y libblas-dev liblapack-dev && \\
#RUN apt-get install -y libhdf5-dev

# Install PETSC
RUN  git clone https://gitlab.com/petsc/petsc /bin/petsc
WORKDIR /bin/petsc
RUN  git checkout v3.11.3 && \
     ./configure --CFLAGS='-O3' --CXXFLAGS='-O3' --FFLAGS='-O3' --with-debugging=no --download-mpich=yes --download-hdf5=yes --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes --download-cmake=yes
ENV PETSC_DIR /bin/petsc
ENV PETSC_ARCH arch-linux-c-opt
WORKDIR /bin/petsc
RUN  make all

# Install PFLOTRAN
WORKDIR /bin
RUN  git clone https://bitbucket.org/pflotran/pflotran
WORKDIR /bin/pflotran/src/pflotran
RUN  make pflotran

# Install python package
RUN pip install h5py
RUN pip install matplotlib

# -----------------------------------------

COPY ./ /kb/module
RUN mkdir -p /kb/module/work
RUN chmod -R a+rw /kb/module

WORKDIR /kb/module

RUN make all

ENTRYPOINT [ "./scripts/entrypoint.sh" ]

CMD [ ]
