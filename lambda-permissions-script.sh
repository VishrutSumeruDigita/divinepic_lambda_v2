#!/bin/bash

# Create the IAM policy using cat EOF
cat << 'EOF' > lambda-permissions-hotfix.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:HeadBucket"
      ],
      "Resource": [
        "arn:aws:s3:::divinepic-test",
        "arn:aws:s3:::divinepic-test/*"
      ]
    },
    {
      "Sid": "VPCNetworkAccess",
      "Effect": "Allow", 
      "Action": [
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface",
        "ec2:AttachNetworkInterface",
        "ec2:DetachNetworkInterface"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Get the Lambda role name
ROLE_NAME=$(aws cloudformation describe-stack-resources \
  --stack-name FastapiModelServingStack \
  --region ap-south-1 \
  --query 'StackResources[?LogicalResourceId==`fastapimodelservingendpointServiceRole`].PhysicalResourceId' \
  --output text)

echo "Found Lambda role: $ROLE_NAME"

# Apply the policy
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "S3ElasticsearchAccess" \
  --policy-document file://lambda-permissions-hotfix.json

echo "✅ Policy applied successfully!"
echo "🚀 Test your embeddings now!" 