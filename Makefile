.PHONY: download


# Download raw datasets (by Udacity)
download: SHELL:=/bin/bash
download:
	@if [[ -f "data/object-detection-crowdai.tar.gz" ]]; then																																							\
		echo "Data exists";																																																									\
		if [[ ! -d "object-detection-crowdai" ]]; then																																											\
			tar xvf data/object-detection-crowdai.tar.gz;																																											\
		fi;																																																																	\
	else																																																																	\
		mkdir -p data;																																																											\
		wget -O data/object-detection-crowdai.tar.gz "https://s3.amazonaws.com/udacity-sdc/annotations/object-detection-crowdai.tar.gz";		\
		tar xvf data/object-detection-crowdai.tar.gz;																																												\
	fi;																																																																		\
	if [[ ! -f "data/labels_crowdai.csv" ]]; then																																													\
		wget -O data/labels_crowdai.csv "https://raw.githubusercontent.com/udacity/self-driving-car/master/annotations/labels_crowdai.csv";	\
	fi


# Generate training images
generate:
	python data_utils.py


cleaner:
	rm -rf data
	rm -rf object-detection-crowdai
