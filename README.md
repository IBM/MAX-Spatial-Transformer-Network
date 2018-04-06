# IBM Code Model Asset Exchange: Spatial Transformer

This repository contains code to train and score a Spatial Transformer Network on [IBM Watson Machine Learning](https://www.ibm.com/cloud/machine-learning). This model is part of the [IBM Code Model Asset Exchange](https://developer.ibm.com/code/exchanges/models/).

> A Spatial Transformer Network allows the spatial manipulation of data within the network.

<div align="center">
  <img width="600px" src="spatial-transformer.png"><br><br>
</div>

>  It can be inserted into existing convolutional architectures, giving neural networks the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process[1].

## How to use

```python
transformer(U, theta, out_size)
```

#### Parameters

```

    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network
```

#### Notes
To initialize the network to the identity transform init ``theta`` to :

```python
identity = np.array([[1., 0., 0.],
                    [0., 1., 0.]])
identity = identity.flatten()
theta = tf.Variable(initial_value=identity)
```

# Quickstart

## Prerequisites

* This experiment requires a provisioned instance of IBM Watson Machine Learning service.

### Setup an IBM Cloud Object Storage (COS) account
- Create an IBM Cloud Object Storage account if you don't have one (https://www.ibm.com/cloud/storage)
- Create credentials for either reading and writing or just reading
	- From the bluemix console page (https://console.bluemix.net/dashboard/apps/), choose `Cloud Object Storage`
	- On the left side, click the service credentials
	- Click on the `new credentials` button to create new credentials
	- In the `Add New Credentials` popup, use this parameter `{"HMAC":true}` in the `Add Inline Configuration...`
	- When you create the credentials, copy the `access_key_id` and `secret_access_key` values.
	- Make a note of the endpoint url
		- On the left side of the window, click on `Endpoint`
		- Copy the relevant public or private endpoint. [I choose the us-geo private endpoint].
- In addition setup your [AWS S3 command line](https://aws.amazon.com/cli/) which can be used to create buckets and/or add files to COS.
   - Export AWS_ACCESS_KEY_ID with your COS `access_key_id` and AWS_SECRET_ACCESS_KEY with your COS `secret_access_key`

### Setup IBM CLI & ML CLI

- Install [IBM Cloud CLI](https://console.bluemix.net/docs/cli/reference/bluemix_cli/get_started.html#getting-started)
  - Login using `bx login` or `bx login --sso` if within IBM
- Install [ML CLI Plugin](https://dataplatform.ibm.com/docs/content/analyze-data/ml_dlaas_environment.html)
  - After install, check if there is any plugins that need update
    - `bx plugin update`
  - Make sure to setup the various environment variables correctly:
    - `ML_INSTANCE`, `ML_USERNAME`, `ML_PASSWORD`, `ML_ENV`

## Training the model

The `train.sh` utility script will deploy the experiment to WML and start the training as a `training-run`

```
train.sh
```

After the train is started, it should print the training-id that is going to be necessary for steps below

```
Starting to train ...
OK
Model-ID is 'training-GCtN_YRig'
```

### Monitor the  training run

- To list the training runs - `bx ml list training-runs`
- To monitor a specific training run - `bx ml show training-runs <training-id>`
- To monitor the output (stdout) from the training run - `bx ml monitor training-runs <training-id>`
- This will print the first couple of lines, and may time out.


### Save and deploy the model after completion

Save the model, when the training run has successfully completed and deploy it for scoring.
- `bx ml store training-runs <training-id>`
	- This should give you back a *model-id*
- `bx ml deploy <model-id> 'spatial-training-deployment'`
	- This should give you a *deployment-id*

## Scoring the model

- Update `modelId` and `deploymentId` on scoring-payload.json
- Score the model with `bx ml score scoring-payload.json`

```
bx ml score scoring-payload.json
Fetching scoring results for the deployment '14f98de1-bc60-4ece-b9f2-3e0c1528c778' ...
{"values": [1]}

OK
Score request successful
```



## Licenses

| Component | License | Link  |
| ------------- | --------  | -------- |
| This repository | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](LICENSE) |
| Model Code (3rd party) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [TensorFlow Models](https://github.com/tensorflow/models/blob/master/LICENSE)|
|Data|[MIT](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/LICENSE)|[Cluttered MNIST ](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/data/mnist_sequence1_sample_5distortions5x5.npz)|

## References

[1] Jaderberg, Max, et al. ["Spatial Transformer Networks"](https://arxiv.org/pdf/1506.02025) arXiv preprint arXiv:1506.02025 (2015)
