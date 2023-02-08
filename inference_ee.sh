gcloud ai-platform local predict --model-dir gs://lyy-shafts/STL/check_pt_senet_100m_TF \
  --json-instances testSample/sample.json \
  --framework tensorflow