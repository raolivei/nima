# Example: Kubernetes Documentation

## What is Kubernetes?

Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications.

### Key Concepts

**Pods**: The smallest deployable units in Kubernetes. A pod encapsulates one or more containers.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
    - name: nginx
      image: nginx:latest
      ports:
        - containerPort: 80
```

**Deployments**: Manages the desired state of your application.

```bash
kubectl create deployment nginx --image=nginx:latest
kubectl scale deployment nginx --replicas=3
```

**Services**: Expose your pods to network traffic.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: LoadBalancer
```

### Common Commands

Deploy an application:

```bash
$ kubectl apply -f deployment.yaml
$ kubectl get pods
$ kubectl logs pod-name
```

Scale your deployment:

```bash
$ kubectl scale deployment my-app --replicas=5
```

### Troubleshooting

When a pod fails to start, check:

1. Pod status: `kubectl describe pod pod-name`
2. Container logs: `kubectl logs pod-name`
3. Events: `kubectl get events`

## Terraform Integration

Terraform can provision Kubernetes clusters:

```hcl
resource "kubernetes_deployment" "example" {
  metadata {
    name = "terraform-example"
  }

  spec {
    replicas = 3

    selector {
      match_labels = {
        app = "MyApp"
      }
    }

    template {
      metadata {
        labels = {
          app = "MyApp"
        }
      }

      spec {
        container {
          image = "nginx:1.21"
          name  = "example"
        }
      }
    }
  }
}
```

## Best Practices

1. **Use namespaces**: Organize resources logically
2. **Resource limits**: Always set CPU and memory limits
3. **Health checks**: Implement liveness and readiness probes
4. **ConfigMaps**: Store configuration separately from code
5. **Secrets**: Never hardcode sensitive data

### Example with Resource Limits

```yaml
resources:
  requests:
    memory: "64Mi"
    cpu: "250m"
  limits:
    memory: "128Mi"
    cpu: "500m"
```

## Docker vs Kubernetes

**Docker** is a containerization platform that packages applications and their dependencies.

**Kubernetes** orchestrates multiple containers across multiple machines.

Think of Docker as building the shipping containers, and Kubernetes as managing the entire shipping fleet.
