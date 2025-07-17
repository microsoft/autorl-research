for i in {1..24}; do
    PYTHONPATH=$PYTHONPATH:/scratch/amlt_code PYTHONUNBUFFERED=1 python rag_agent.py 2>&1 | tee -a "./output/agent${i}.log" &
done