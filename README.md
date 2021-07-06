# Urban 3D Challenge

High-resolution satellite imagery is changing our understanding of the world around us, as well as the way we as humans interact with our planet. However, raw images do little more than pique our interest unless we can superimpose a layer that actually identifies real objects. Reliable labeling of building footprints based on satellite imagery is one of the first and most challenging steps in producing accurate 3D models. While automated algorithms continue to improve, significant manual effort is still required to ensure geospatial accuracy and acceptable quality. Improved automation is required to enable more rapid response to major world events such as humanitarian and disaster response. 3D height data can help improve automated building footprint detection performance, and capabilities for providing this data on a global scale are now emerging. In this challenge, contestants used 2D and 3D imagery generated from commercial satellite imagery along with state of the art machine learning techniques to provide high quality, automated building footprint detection performance over large areas.

TopCoder helped execute this challenge and is now providing winning solutions as open source software. See below for more information on how to obtain the software and pre-trained models. 

This challenge also published a large-scale dataset containing 2D orthrorectified RGB and 3D Digital Surface Models and Digital Terrain Models generated from commercial satellite imagery covering over 360 km of terrain and containing roughly 157,000 annotated building footprints. All imagery products are provided at 50 cm ground sample distance (GSD). This unique 2D/3D large scale dataset provides researchers an opportunity to utilize machine learning techniques to further improve state of the art performance. Information on how to obtain this dataset is found below. 

See more background information about the challenge [here.](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&compid=57607)

# Winning solutions and algorithm descriptions

The top six (6) winning solutions are provided as open source as part of this Github repository. Information on how to setup and use the software can be found in the [Software Users Guide](https://github.com/topcoderinc/Urban3d/blob/master/Urban%203D%20Challenge%20Software%20User%20Guide.docx). 

NOTES: 
* Some subfolders contain submission as a single zip file, others have a subfolder where files are unzipped. This had to be done this way because github doesn't support files larger than 100MB.
* Due to size limitations, the pre-trained models for the solutions are not distributable on Github. Instead, they can be found along with the datasets [here](https://spacenetchallenge.github.io/datasets/Urban_3D_Challenge_summary.html). 


## Final scores of the top six (6) winning solutions using the sequestered dataset
| Handle | score |
| --- | --- |
| albu | 855174.32 |
| selim_sef | 852711.49 |
| cannab | 850676.75 |
| alina.marcu | 850520.81 |
| kylelee | 849570.67 |
| ZFTurbo | 847606.45 |

# Urban 3D Challenge Datasets 

The complete datasets used for the Urban 3D Challenge can be found on the [SpaceNet AWS page](https://spacenet.ai/the-ussocom-urban-3d-competition/). Please refer to documentation on that website for instructions on how to download the data. 

**PLEASE NOTE: The data on AWS S3 has moved from s3://spacenet-dataset/Urban_3D_Challenge/ to s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/. This change has not yet been reflected on the SpaceNet AWS Urban 3D Challenge page. We are working with SpaceNet to have this corrected.**

# Algorithm Evaluation

A custom visualizer is provided with the algorithms in this github repository. Please refer to the README HTML page inside that folder for instructions on how to use the visualizer to evaluate algorithm performance. 

# Publications

Discriptions of the publicly released benchmark dataset as well as additional analysis of the winning solutions can be found in the following publications: 
* H. Goldberg, M. Brown, and S. Wang, A Benchmark for Building Footprint Classification Using Orthorectified RGB Imagery and Digital Surface Models from Commercial Satellites, 46th Annual IEEE Applied Imagery Pattern Recognition Workshop, Washington, D.C, 2017.
* H. Goldberg, S. Wang, M. Brown, and G. Christie. Urban 3D Challenge: Building Footprint Detection Using Orthorectified Imagery and Digital Surface Models from Commercial Satellites. In Proceedings SPIE Defense and Commercial Sensing: Geospatial Informatics and Motion Imagery Analytics VIII, Orlando, Florida, USA, 2018.

# Questions?

For questions about the Urban 3D Challenge, please contact pubgeo(at)jhuapl(dot)edu. 



