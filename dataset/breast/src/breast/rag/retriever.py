def retrieve_context(vectorestore, query, k=3):
    docs = vectorestore.similarity(query,k=k)
    return '\n\n'.join([d.page_content for d in docs])