exec: task3.cpp
	g++ -o $@ $(INCLUDES) $< $(-L/path/to/my/openCV/lib ) -lopencv_core  -lopencv_imgcodecs -lopencv_highgui /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2 -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_objdetect 
.PHONY: run

run: exec
	@./$<
