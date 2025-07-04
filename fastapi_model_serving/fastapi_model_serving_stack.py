import os
from aws_cdk import (
    Size,
    Duration,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
)
from constructs import Construct

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class FastapiModelServingStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        fastapi_model_endpoint_lambda = _lambda.DockerImageFunction(
            self,
            "fastapi_model_serving_endpoint",
            function_name="fastapi_model_serving_endpoint_docker",
            architecture=_lambda.Architecture.X86_64,
            code=_lambda.DockerImageCode.from_image_asset(
                os.path.join(DIR_PATH, "..", "model_endpoint", "docker")
            ),
            timeout=Duration.seconds(900),  # 15 minutes for face processing
            ephemeral_storage_size=Size.mebibytes(10240),  # 10GB for models and temp files
            memory_size=3008,  # 3008MB - Maximum allowed by Lambda
        )

        apigateway.LambdaRestApi(
            self,
            "docker_model_serving_endpoint",
            handler=fastapi_model_endpoint_lambda,
            proxy=True,
        )
