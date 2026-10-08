# Shared settings of the experiment scripts. PY selects the interpreter; stores and results are JSON Lines.
PY=${PY:-python}
RUN="$PY -m expreuse.experiments.run"
STORES=${STORES:-stores}
RESULTS=${RESULTS:-results}
mkdir -p "$STORES" "$RESULTS"
producer() {  # the name a consumer's traces are stored under (config.CONSUMERS)
  $PY -c "from expreuse import config; print(config.CONSUMERS['$1']['producer'])"
}
