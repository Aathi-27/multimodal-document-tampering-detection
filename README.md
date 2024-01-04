## Document Tampering Detection

Given the objective of detecting document tampering, we will employ a computer vision model hosted on SageMaker for our image forgery detection solution. This model operates by receiving a testing image as input and generates a likelihood prediction of forgery as its output.

Given the expansive realm of image forgery detection, we'll utilize the Error Level Analysis (ELA) algorithm as an illustrative method for detecting forgeries. ELA functions by identifying discrepancies in compression levels within an image. ELA's underlying premise assumes that the input images are in JPEG format, known for its lossy compression.

While powerful, ELA has some limitations in identifying certain types of more subtle manipulation. As future work, the model could be enhanced by incorporating additional forensic techniques into training and leveraging larger, more diverse datasets. Overall, this demonstrates how deep learning and AWS services can be leveraged to build impactful solutions that boost efficiency, reduce risk, and prevent fraud.

To use the content, clone the repository into your Amazon SageMaker Studio environment and follow the instructions in the notebooks

tampering_detection_training.ipynb - This notebook walks you through the steps to train a CNN model for image tampering detection using the Error Level Analysis (ELA) technique

tampering_detection_model_deploy.ipynb - This notebook walks you through the steps to deploy the trained CNN model for document tampering detection to Amazon SageMaker Inference as a real-time API

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

