FROM jbirgio/neuron:7.7

#Expose port 5920 to view display using VNC Viewer
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y git x11vnc xvfb openbox menu
EXPOSE 5920
ENV DISPLAY=:20

# Run simple window manager on startup
RUN echo 'openbox-session 2> /dev/null || true &' >> /root/.bashrc

RUN conda install pandas=1.0.3 llvmlite=0.34.0 -y
RUN python -m pip install pip==9.0.3
RUN pip install pysqlite3==0.4.3 jupyterlab==0.31.5 blenderneuron==2.0.4 Cython==0.28.1 jsonpickle==0.9.6 matplotlib==2.2.2 mpi4py==3.0.0 neo==0.6.1 sciunit neuronunit numpy==1.18.5 pandas==1.0.3 peewee==3.13.3 quantities==0.12.1 scipy==1.4.1 deap==1.2.2 LFPsimpy==0.1.1

# Compile mod files
#WORKDIR "/OlfactoryBulb/prev_ob_models"
#RUN ./compile_mod.sh
#WORKDIR "/OlfactoryBulb"

# Start container terminal model dir
#RUN echo 'cd /OlfactoryBulb' >> /root/.bashrc
EXPOSE 8888

# Create a vnc server to share the screen and jupyter lab to view the notebook
CMD ["sh", "-c", "\
     x11vnc -nopw -create -rfbport 5920 -forever & disown;\
     jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser;\
"]
