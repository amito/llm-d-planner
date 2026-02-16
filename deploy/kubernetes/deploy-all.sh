#!/bin/bash

oc apply -f deploy/kubernetes/namespace.yaml \
         -f deploy/kubernetes/secrets.yaml \
         -f deploy/kubernetes/postgres.yaml \
         -f deploy/kubernetes/ollama.yaml \
         -f deploy/kubernetes/backend.yaml \
         -f deploy/kubernetes/ui.yaml \
         -f deploy/kubernetes/route.yaml
