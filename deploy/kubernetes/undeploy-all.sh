#!/bin/bash
set -e

NAMESPACE="${PLANNER_NAMESPACE:-planner}"
CLUSTER_ADMIN="${CLUSTER_ADMIN:-true}"
LABEL="app.kubernetes.io/part-of=planner"

echo "Removing Planner from namespace: ${NAMESPACE}..."

oc delete deployment,service,route,job,secret,configmap,serviceaccount,networkpolicy,pvc \
  -l "$LABEL" -n "$NAMESPACE" --ignore-not-found

if [ "$CLUSTER_ADMIN" = "true" ]; then
  echo "Removing cluster-scoped RBAC..."
  oc delete clusterrole,clusterrolebinding -l "$LABEL" --ignore-not-found
else
  echo "Skipping cluster-scoped RBAC (CLUSTER_ADMIN=false)"
fi

echo "Undeploy complete."
