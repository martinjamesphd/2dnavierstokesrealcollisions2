CC = gcc
CFLAG = -Wall -fopenmp
look = -L/export/share/fftw-2.1.5/lib -L/home/ssray/Martin/fftw-2.1.5/threads/.libs -I/export/share/fftw-2.1.5/include -I//home/ssray/Martin/fftw-2.1.5/threads
link = -lrfftw_threads -lfftw_threads -lrfftw -lfftw -lm -lpthread
all = main.o time_march.o init_flow.o find_vel.o particle.o

ns2d : $(all)
	$(CC) -o $@ $(all) $(look) $(link) $(CFLAG)
	
main.o : main.c
	$(CC) -c -o $@ main.c $(look) $(CFLAG)
	
time_march.o : time_march.c
	$(CC) -c -o $@ time_march.c $(look) $(CFLAG)
	
init_flow.o : init_flow.c
	$(CC) -c -o $@ init_flow.c $(look) $(CFLAG)
	
find_vel.o : find_vel.c
	$(CC) -c -o $@ find_vel.c $(look) $(CFLAG)
	
particle.o : particle.c
	$(CC) -c -o $@ particle.c $(look) $(CFLAG)

clean:
	rm *.o

clean_data:
	rm field/* spectra/* particle/*
