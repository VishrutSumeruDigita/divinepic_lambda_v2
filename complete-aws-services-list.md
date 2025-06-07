# Complete AWS Services & Permissions List for CDK Deployment

## Core CDK Services (Bootstrap & Deployment)

### 1. **CloudFormation**
- `cloudformation:*` (All permissions)
- **Why**: CDK uses CloudFormation as the deployment engine

### 2. **S3**
- `s3:CreateBucket`, `s3:DeleteBucket`
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`
- `s3:GetObjectVersion`, `s3:DeleteObjectVersion`
- `s3:ListBucket`, `s3:ListBucketVersions`, `s3:ListAllMyBuckets`
- `s3:GetBucketLocation`, `s3:GetBucketAcl`, `s3:PutBucketAcl`
- `s3:GetBucketPolicy`, `s3:PutBucketPolicy`, `s3:DeleteBucketPolicy`
- `s3:GetBucketVersioning`, `s3:PutBucketVersioning`
- `s3:GetBucketPublicAccessBlock`, `s3:PutBucketPublicAccessBlock`
- `s3:GetBucketEncryption`, `s3:PutBucketEncryption`
- `s3:GetBucketTagging`, `s3:PutBucketTagging`, `s3:DeleteBucketTagging`
- `s3:GetBucketOwnershipControls`, `s3:PutBucketOwnershipControls`
- **Why**: CDK assets storage and deployment artifacts

### 3. **IAM**
- `iam:*` (All permissions needed for role/policy management)
- `iam:PassRole` (specific to Lambda, API Gateway, CloudFormation services)
- **Why**: Creating execution roles for Lambda and service-linked roles

### 4. **SSM Parameter Store**
- `ssm:GetParameter`, `ssm:GetParameters`, `ssm:PutParameter`
- `ssm:DeleteParameter`, `ssm:GetParametersByPath`, `ssm:DescribeParameters`
- **Why**: CDK bootstrap version tracking

## Application-Specific Services

### 5. **Lambda**
- `lambda:CreateFunction`, `lambda:DeleteFunction`, `lambda:UpdateFunctionCode`
- `lambda:UpdateFunctionConfiguration`, `lambda:GetFunction`
- `lambda:ListFunctions`, `lambda:TagResource`, `lambda:UntagResource`
- `lambda:AddPermission`, `lambda:RemovePermission`
- `lambda:GetPolicy`, `lambda:CreateEventSourceMapping`
- `lambda:UpdateEventSourceMapping`, `lambda:DeleteEventSourceMapping`
- **Why**: Your FastAPI model serving endpoint

### 6. **ECR (Elastic Container Registry)**
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`
- `ecr:BatchGetImage`, `ecr:DescribeRepositories`, `ecr:CreateRepository`
- `ecr:BatchDeleteImage`, `ecr:ListImages`, `ecr:DescribeImages`
- `ecr:CompleteLayerUpload`, `ecr:GetRepositoryPolicy`
- `ecr:InitiateLayerUpload`, `ecr:PutImage`, `ecr:UploadLayerPart`
- `ecr:SetRepositoryPolicy`, `ecr:DeleteRepository`
- **Why**: Docker image storage for Lambda function

### 7. **API Gateway**
- `apigateway:*` (All permissions)
- `execute-api:*` (for testing)
- **Why**: REST API for your Lambda function

### 8. **CloudWatch Logs**
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
- `logs:DescribeLogGroups`, `logs:DescribeLogStreams`
- `logs:DeleteLogGroup`, `logs:DeleteLogStream`
- **Why**: Lambda function logging

## Security & Encryption Services

### 9. **KMS**
- `kms:Decrypt`, `kms:Encrypt`, `kms:ReEncrypt*`
- `kms:GenerateDataKey*`, `kms:DescribeKey`
- `kms:CreateGrant`, `kms:ListAliases`, `kms:ListKeys`
- **Why**: Encryption for S3, Lambda environment variables

### 10. **STS (Security Token Service)**
- `sts:AssumeRole`, `sts:GetCallerIdentity`
- **Why**: Role assumption and identity verification

## Additional Services (May be needed)

### 11. **EventBridge (CloudWatch Events)**
- `events:*`
- **Why**: Potential future integrations, CDK sometimes uses this

### 12. **Resource Groups & Tagging**
- `tag:GetResources`, `tag:TagResources`, `tag:UntagResources`
- `resource-groups:*`
- **Why**: Resource organization and cost tracking

### 13. **Application Auto Scaling**
- `application-autoscaling:*`
- **Why**: Potential Lambda concurrency scaling

### 14. **Organizations (if applicable)**
- `organizations:DescribeAccount`, `organizations:DescribeOrganization`
- `organizations:ListAccounts`, `organizations:ListParents`, `organizations:ListRoots`
- **Why**: Account information during deployment

### 15. **Service Catalog (if using CDK Pipelines)**
- `servicecatalog:*`
- **Why**: Advanced CDK deployment patterns

## Services NOT Currently Needed (but common in CDK apps)

- **DynamoDB**: Not used in your current stack
- **SNS/SQS**: Not used in your current stack  
- **VPC/EC2**: Lambda runs in managed VPC
- **RDS**: Not used in your current stack
- **Route53/CloudFront**: Not used in your current stack
- **Secrets Manager**: Not used in your current stack (but recommended for API keys)

## Minimal Required Services (if you want to start small)

If your DevOps team wants to grant minimal permissions first:

1. **CloudFormation** (full access)
2. **S3** (full access)
3. **IAM** (full access)
4. **SSM** (Parameter Store access)
5. **ECR** (full access)
6. **Lambda** (full access)
7. **API Gateway** (full access)
8. **CloudWatch Logs** (full access)
9. **STS** (assume role access)

## Recommended Policy Attachment Method

```bash
# Option 1: Attach AWS Managed Policies (easier but broader permissions)
aws iam attach-user-policy --user-name divinepic-test-user --policy-arn arn:aws:iam::aws:policy/PowerUserAccess

# Option 2: Use the custom policy we created (more secure)
aws iam put-user-policy --user-name divinepic-test-user --policy-name CDKDeploymentPolicy --policy-document file://cdk-permissions-policy.json
```

## Cost Implications

- **Lambda**: Pay per request + duration
- **API Gateway**: Pay per API call
- **ECR**: Pay for storage (few dollars/month for Docker images)
- **S3**: Pay for storage (minimal for CDK assets)
- **CloudWatch Logs**: Pay for log storage and queries

**Estimated monthly cost**: $5-20 for low-medium traffic 