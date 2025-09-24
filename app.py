import streamlit as st
import json
import re
from graphviz import Digraph
from collections import deque, defaultdict
import networkx as nx
from PIL import Image

# --- Core Logic ---

def parse_sql_statements(sql_block):
    if not sql_block: return []
    sql_block = re.sub(r'--.*', '', sql_block).strip()
    return [stmt.strip() + ';' for stmt in sql_block.split(';') if stmt.strip()]

def parse_single_sql(sql_statement):
    if not sql_statement: return set(), None
    sql_statement = sql_statement.upper().replace('\n', ' ')
    inputs, output = set(), None
    output_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+`?("?[\w\d_.-]+"?)`?|INSERT\s+INTO\s+`?("?[\w\d_.-]+"?)`?', sql_statement)
    if output_match:
        output_name = output_match.group(1) or output_match.group(2)
        if output_name: output = output_name.replace('"', '').replace('`', '')
    input_matches = re.findall(r'(?:FROM|JOIN)\s+`?("?[\w\d_.-]+"?)`?\s*(?:AS\s+[\w\d_]+)?', sql_statement)
    for table in input_matches:
        if '<%=' in table:
            table_name = re.search(r'"([\w\d_.-]+)"', table)
            if table_name: inputs.add(table_name.group(1))
        else: inputs.add(table.replace('"', '').replace('`', ''))
    if output and output in inputs: inputs.remove(output)
    return inputs, output

def build_dependency_maps(pipeline_data):
    dependency_map, reverse_dependency_map = {}, {}
    sql_by_task = {}  
    
    tasks = pipeline_data.get('tasks', [])
    for task in tasks:
        if task.get('taskType') == 'TRANSFORMATION':
            task_id = task.get('id')
            sql_block = task.get('transformationStatement', '')
            if sql_block and task_id:
                sql_by_task[task_id] = sql_block
                
                for stmt in parse_sql_statements(sql_block):
                    inputs, output = parse_single_sql(stmt)
                    if output:
                        output_cleaned = output.split('.')[-1]
                        inputs_cleaned = {inp.split('.')[-1] for inp in inputs}
                        dependency_map.setdefault(output_cleaned, []).append({
                            'inputs': inputs_cleaned, 
                            'task_id': task_id,
                            'sql': stmt
                        })
                        for inp in inputs_cleaned:
                            reverse_dependency_map.setdefault(inp, set()).add(output_cleaned)
    
    transformations = pipeline_data.get('transformations', [])
    for trans in transformations:
        task_id = trans.get('id')
        sql_block = trans.get('transformationStatement', '')
        if sql_block and task_id:
            sql_by_task[task_id] = sql_block
            
            for stmt in parse_sql_statements(sql_block):
                inputs, output = parse_single_sql(stmt)
                if output:
                    output_cleaned = output.split('.')[-1]
                    inputs_cleaned = {inp.split('.')[-1] for inp in inputs}
                    dependency_map.setdefault(output_cleaned, []).append({
                        'inputs': inputs_cleaned, 
                        'task_id': task_id,
                        'sql': stmt
                    })
                    for inp in inputs_cleaned:
                        reverse_dependency_map.setdefault(inp, set()).add(output_cleaned)
    
    for extr in pipeline_data.get('tableExtractions', []):
        if not extr.get('disabled'):
            table_name = (extr.get('targetTableName') or extr.get('tableName', '')).upper()
            task_id = extr.get('taskId')
            if table_name and task_id:
                dependency_map.setdefault(table_name, []).append({
                    'inputs': {'SAP_SOURCE'}, 
                    'task_id': task_id
                })
                reverse_dependency_map.setdefault('SAP_SOURCE', set()).add(table_name)
    
    return dependency_map, reverse_dependency_map, sql_by_task

def find_unused_objects(dependency_map, reverse_dependency_map, final_tables):
    unused_tables = []
    orphaned_sources = []
    
    for table in dependency_map.keys():
        if table in final_tables or table == 'SAP_SOURCE':
            continue
        if table not in reverse_dependency_map or not reverse_dependency_map[table]:
            unused_tables.append(table)
    
    for table in reverse_dependency_map.keys():
        if table == 'SAP_SOURCE':
            continue
        if table not in dependency_map and table not in final_tables:
            orphaned_sources.append(table)
    
    return unused_tables, orphaned_sources

def analyze_dependency_chains(dependency_map, reverse_dependency_map, final_tables):
    chain_lengths = {}
    
    def get_chain_length(table, visited=None):
        if visited is None:
            visited = set()
        if table in visited:
            return 0
        visited.add(table)
        
        if table not in dependency_map:
            return 0
        
        max_length = 0
        for source_info in dependency_map[table]:
            for input_table in source_info['inputs']:
                if input_table != 'SAP_SOURCE':
                    length = 1 + get_chain_length(input_table, visited.copy())
                    max_length = max(max_length, length)
        return max_length
    
    for table in final_tables:
        chain_lengths[table] = get_chain_length(table)
    
    return chain_lengths

def search_in_pipeline(search_term, dependency_map, sql_by_task):
    results = {
        'tables': [],
        'sql_matches': []
    }
    
    search_upper = search_term.upper()
    
    for table in dependency_map.keys():
        if search_upper in table:
            results['tables'].append(table)
    
    for task_id, sql in sql_by_task.items():
        if search_upper in sql.upper():
            results['sql_matches'].append({
                'task_id': task_id,
                'sql_snippet': sql[:200] + '...' if len(sql) > 200 else sql
            })
    
    return results

def get_extraction_summary(pipeline_data, all_tasks):
    extractions_by_table = defaultdict(list)
    
    for extr in pipeline_data.get('tableExtractions', []):
        table_name = extr.get('targetTableName') or extr.get('tableName', 'UNKNOWN')
        task_id = extr.get('taskId')
        task_name = all_tasks.get(task_id, {}).get('name', f'Task {task_id}' if task_id else 'No Task')
        
        extraction_info = {
            'disabled': extr.get('disabled', False),
            'mode': 'DELTA' if extr.get('changeDateColumn') else 'FULL',
            'filter': extr.get('filterDefinition', ''),
            'delta_filter': extr.get('deltaFilterDefinition', ''),
            'change_date_column': extr.get('changeDateColumn', ''),
            'task_name': task_name
        }
        extractions_by_table[table_name.upper()].append(extraction_info)
    
    return extractions_by_table

def get_final_tables_and_aliases(datamodel_data):
    final_tables, alias_map = set(), {}
    for model in datamodel_data.get('dataModels', []):
        for table in model.get('tables', []):
            real_name = table.get('name', '').upper()
            alias = (table.get('aliasOrName') or table.get('name', '')).upper()
            if real_name and alias:
                final_tables.add(alias)
                alias_map[alias] = real_name
    return final_tables, alias_map

def trace_lineage(start_nodes, dep_map, reverse_dep_map, direction='upstream', depth=5):
    nodes_to_render, edges_to_render = set(start_nodes), set()
    queue = deque([(node, 0) for node in start_nodes])
    processed = set()
    
    while queue:
        table, current_depth = queue.popleft()
        if table in processed or current_depth >= depth:
            continue
        processed.add(table)
        
        if direction == 'upstream' and table in dep_map:
            for info in dep_map[table]:
                task_id = info['task_id']
                nodes_to_render.add(task_id)
                for input_table in info['inputs']:
                    nodes_to_render.add(input_table)
                    edges_to_render.add((input_table, task_id))
                    edges_to_render.add((task_id, table))
                    if input_table not in processed:
                        queue.append((input_table, current_depth + 1))
        elif direction == 'downstream' and table in reverse_dep_map:
            for child_table in reverse_dep_map.get(table, []):
                if child_table in dep_map:
                    for info in dep_map[child_table]:
                        if table in info['inputs']:
                            task_id = info['task_id']
                            nodes_to_render.add(task_id)
                            nodes_to_render.add(child_table)
                            edges_to_render.add((table, task_id))
                            edges_to_render.add((task_id, child_table))
                            if child_table not in processed:
                                queue.append((child_table, current_depth + 1))
    
    return nodes_to_render, edges_to_render

def generate_graph_data(nodes, edges, pipeline_data, final_tables, alias_map, title, output_format='pdf'):
    if not nodes:
        st.warning("No nodes available to generate the graph.")
        return None
    
    try:
        dot = Digraph(engine='dot')
        dot.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='1.0', 
                 label=title, labelloc='t', fontsize='24', fontname="Helvetica", dpi='300')
        
        styles = {
            "final_table": {'shape': 'folder', 'style': 'filled', 'fillcolor': '#D5E8D4', 'color': '#82b366'},
            "intermediate_table": {'shape': 'folder', 'style': 'filled', 'fillcolor': '#DAE8FC'},
            "task": {'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#FFF2CC'},
            "source_system": {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F8CECC'}
        }
        
        all_tasks = {}
        for t in pipeline_data.get('tasks', []):
            if 'id' in t:
                all_tasks[t['id']] = t
        for t in pipeline_data.get('transformations', []):
            if 'id' in t:
                all_tasks[t['id']] = t
        
        for node_id in nodes:
            node_id_str = str(node_id)
            if node_id == 'SAP_SOURCE':
                dot.node(node_id_str, 'SAP Source System', **styles['source_system'])
            elif node_id in all_tasks:
                task_name = all_tasks[node_id].get('name', 'Unnamed Task')
                dot.node(node_id_str, f"Task: {task_name}", **styles['task'])
            else:
                if node_id in final_tables or node_id in alias_map.values():
                    dot.node(node_id_str, node_id_str, **styles['final_table'])
                else:
                    dot.node(node_id_str, node_id_str, **styles['intermediate_table'])
        
        for source, target in edges:
            dot.edge(str(source), str(target))
        
        return dot.pipe(format=output_format)
    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
        return None

def offer_graph_download(nodes, edges, pipeline_data, final_tables, alias_map, title):
    st.subheader(title)
    
    st.write(f"Number of nodes: {len(nodes)}")
    st.write(f"Number of edges: {len(edges)}")
    
    with st.spinner("Generating PDF..."):
        pdf_data = generate_graph_data(nodes, edges, pipeline_data, final_tables, alias_map, title, output_format='pdf')
        if pdf_data:
            st.download_button(
                label="Download as High-Quality PDF",
                data=pdf_data,
                file_name=f"{title.replace(' ', '_').lower()}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("No data available to generate a graph for this selection.")

def display_graph_preview_and_download(nodes, edges, pipeline_data, final_tables, alias_map, title):
    st.subheader(title)
    st.write(f"Number of nodes: {len(nodes)}")
    st.write(f"Number of edges: {len(edges)}")
    with st.spinner("Generating graph..."):
        try:
            Image.MAX_IMAGE_PIXELS = 500000000
            preview_png = generate_graph_data(nodes, edges, pipeline_data, final_tables, alias_map, title, output_format='png')
            if preview_png:
                st.image(preview_png)
                st.markdown("---")
            pdf_data = generate_graph_data(nodes, edges, pipeline_data, final_tables, alias_map, title, output_format='pdf')
            if pdf_data:
                st.download_button(
                    label="Download as High-Quality PDF",
                    data=pdf_data,
                    file_name=f"{title.replace(' ', '_').lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF for download.")
        except Image.DecompressionBombError:
            st.error("The generated graph is too large to display as an image. Downloading the PDF version instead.")
            pdf_data = generate_graph_data(nodes, edges, pipeline_data, final_tables, alias_map, title, output_format='pdf')
            if pdf_data:
                st.download_button(
                    label="Download as High-Quality PDF",
                    data=pdf_data,
                    file_name=f"{title.replace(' ', '_').lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error displaying graph: {str(e)}")

# --- Streamlit App Main Function ---
def main():
    st.set_page_config(page_title="Data Lineage Explorer", layout="wide")
    st.title("Data Lineage & Refactoring Explorer")

    with st.sidebar:
        st.header("Upload Files")
        pipeline_file = st.file_uploader("Upload Pipeline JSON", type="json")
        datamodel_file = st.file_uploader("Upload Data Model JSON", type="json")

    if not (pipeline_file and datamodel_file):
        st.info("Please upload your Pipeline and Data Model JSON files to begin.")
        return

    try:
        pipeline_data = json.load(pipeline_file)
        datamodel_data = json.load(datamodel_file)
        dependency_map, reverse_dependency_map, sql_by_task = build_dependency_maps(pipeline_data)
        final_tables, alias_map = get_final_tables_and_aliases(datamodel_data)
        
        Full_G = nx.DiGraph()
        for child, sources in dependency_map.items():
            for info in sources:
                task = info['task_id']
                for parent in info['inputs']:
                    Full_G.add_edge(parent, task)
                Full_G.add_edge(task, child)
        global components
        components = [c for c in sorted(nx.weakly_connected_components(Full_G), key=len, reverse=True)]
        
        all_tasks = {}
        for t in pipeline_data.get('tasks', []):
            if 'id' in t:
                all_tasks[t['id']] = t
        for t in pipeline_data.get('transformations', []):
            if 'id' in t:
                all_tasks[t['id']] = t

    except Exception as e:
        st.error(f"Error processing uploaded files: {e}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Lineage Explorer",
        "Pipeline Explorer",
        "Pipeline Analysis",
        "Search & SQL Viewer",
        "Extraction Summary"
    ])

    with tab1:
        st.header("Lineage Explorer")
        all_tables = sorted(list(dependency_map.keys() | reverse_dependency_map.keys()))
        col1, col2 = st.columns(2)
        with col1:
            selected_table = st.selectbox("Select a table to trace:", all_tables)
            trace_depth = st.slider("Trace Depth", 1, 10, 5)
        with col2:
            trace_direction = st.radio("Trace Direction:", ("Upstream (Dependencies)", "Downstream (Impact)"))
        
        if st.button("Trace Table Lineage", use_container_width=True):
            direction = 'upstream' if 'Upstream' in trace_direction else 'downstream'
            nodes, edges = trace_lineage([selected_table], dependency_map, reverse_dependency_map, direction, trace_depth)
            title = f"{direction.title()} Lineage for Table: {selected_table}"
            display_graph_preview_and_download(nodes, edges, pipeline_data, final_tables, alias_map, title)

    with tab2:
        st.header("Pipeline Explorer")
        st.markdown("View the entire pipeline graph or inspect its individual disconnected components.")
        
        st.write(f"Full graph nodes: {len(Full_G.nodes())}")
        st.write(f"Full graph edges: {len(Full_G.edges())}")
        
        # Full graph section
        st.subheader("Full Pipeline Graph")
        if st.button("Generate Full Graph View", use_container_width=True):
            selected_nodes = list(Full_G.nodes())
            subgraph = Full_G.subgraph(selected_nodes)
            title = "Full Pipeline Graph"
            offer_graph_download(subgraph.nodes(), subgraph.edges(), pipeline_data, final_tables, alias_map, title)
        
        # Individual subgraphs
        st.subheader("Individual Subgraphs")
        for i, component in enumerate(components):
            st.markdown(f"**Subgraph {i+1} ({len(component)} nodes)**")
            if st.button(f"Generate Subgraph {i+1} View", key=f"subgraph_{i}", use_container_width=True):
                selected_nodes = list(component)
                subgraph = Full_G.subgraph(selected_nodes)
                title = f"Subgraph {i+1}"
                display_graph_preview_and_download(subgraph.nodes(), subgraph.edges(), pipeline_data, final_tables, alias_map, title)

    with tab3:
        st.header("Pipeline Analysis")
        st.subheader("Dependency Chain Analysis")
        chain_lengths = analyze_dependency_chains(dependency_map, reverse_dependency_map, final_tables)
        
        if chain_lengths:
            max_chain = max(chain_lengths.items(), key=lambda x: x[1])
            avg_chain = sum(chain_lengths.values()) / len(chain_lengths) if chain_lengths else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Longest Chain", f"{max_chain[1]} levels")
                st.caption(f"Table: {max_chain[0]}")
            with col2:
                st.metric("Average Chain Length", f"{avg_chain:.1f} levels")
            with col3:
                st.metric("Tables Analyzed", len(chain_lengths))
            
            long_chains = sorted(chain_lengths.items(), key=lambda x: x[1], reverse=True)[:5]
            if long_chains:
                st.markdown("Longest Dependency Chains (refactoring candidates):")
                for table, length in long_chains:
                    st.write(f"- {table}: {length} levels")
        
        st.markdown("---")
        
        st.subheader("Unused Objects (Can be removed)")
        unused_tables, orphaned_sources = find_unused_objects(dependency_map, reverse_dependency_map, final_tables)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unused Intermediate Tables", len(unused_tables))
            if unused_tables:
                with st.expander("View unused tables"):
                    for table in unused_tables:
                        st.write(f"- {table}")
        
        with col2:
            st.metric("Orphaned Source Tables", len(orphaned_sources))
            if orphaned_sources:
                with st.expander("View orphaned tables"):
                    for table in orphaned_sources:
                        st.write(f"- {table}")
        
        st.markdown("---")
        
        st.subheader("Convergence Points")
        convergence_points = {table: sources for table, sources in dependency_map.items() if len(sources) > 1}
        
        if convergence_points:
            sorted_convergence = sorted(convergence_points.items(), key=lambda x: len(x[1]), reverse=True)
            st.write(f"Found {len(sorted_convergence)} tables with multiple sources:")
            for table, sources in sorted_convergence:
                with st.expander(f"{table} (Generated by {len(sources)} sources)"):
                    for source_info in sources:
                        task_id = source_info['task_id']
                        task_name = all_tasks.get(task_id, {}).get('name', 'Unknown Task')
                        st.write(f"- Task: {task_name}")
                        if 'sql' in source_info and source_info['sql']:
                            st.code(source_info['sql'][:300], language='sql')
        else:
            st.info("No convergence points found.")

    with tab4:
        st.header("Search & SQL Viewer")
        search_term = st.text_input("Search for table names or SQL patterns:")
        if search_term:
            results = search_in_pipeline(search_term, dependency_map, sql_by_task)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Tables Found")
                if results['tables']:
                    for table in results['tables']:
                        st.write(f"- {table}")
                else:
                    st.info("No tables found matching your search.")
            
            with col2:
                st.subheader("SQL Matches")
                if results['sql_matches']:
                    for match in results['sql_matches'][:5]:
                        task_name = all_tasks.get(match['task_id'], {}).get('name', f"Task {match['task_id']}")
                        st.write(f"Task: {task_name}")
                        st.code(match['sql_snippet'], language='sql')
                else:
                    st.info("No SQL matches found.")
        
        st.markdown("---")
        
        st.subheader("SQL Statement Viewer")
        if sql_by_task:
            task_options = {}
            for task_id, sql in sql_by_task.items():
                task_name = all_tasks.get(task_id, {}).get('name', f'Task {task_id}')
                task_options[task_name] = (task_id, sql)
            
            selected_task_name = st.selectbox("Select a transformation:", sorted(task_options.keys()))
            
            if selected_task_name:
                task_id, sql = task_options[selected_task_name]
                st.code(sql, language='sql')
                
                st.download_button(
                    label="Download SQL",
                    data=sql,
                    file_name=f"{selected_task_name.replace(' ', '_')}.sql",
                    mime="text/plain"
                )
        else:
            st.info("No SQL transformations found in the pipeline.")

    with tab5:
        st.header("Extraction Summary")
        extractions_by_table = get_extraction_summary(pipeline_data, all_tasks)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_extractions = sum(len(extrs) for extrs in extractions_by_table.values())
            st.metric("Total Extractions", total_extractions)
        with col2:
            disabled_count = sum(1 for extrs in extractions_by_table.values() for extr in extrs if extr['disabled'])
            st.metric("Disabled Extractions", disabled_count)
        with col3:
            delta_count = sum(1 for extrs in extractions_by_table.values() for extr in extrs if extr['mode'] == 'DELTA')
            st.metric("Delta Loads", delta_count)
        
        col1, col2 = st.columns(2)
        with col1:
            show_disabled = st.checkbox("Show disabled extractions", value=False)
        with col2:
            filter_mode = st.radio("Filter by mode:", ["All", "DELTA", "FULL"], horizontal=True)
        
        st.subheader("Extraction Details by Table")
        for table_name in sorted(extractions_by_table.keys()):
            extrs = extractions_by_table[table_name]
            filtered_extrs = [
                extr for extr in extrs
                if (show_disabled or not extr['disabled']) and (filter_mode == "All" or extr['mode'] == filter_mode)
            ]
            if not filtered_extrs:
                continue
            with st.expander(f"Table: {table_name} ({len(filtered_extrs)} extractions)"):
                for extr in filtered_extrs:
                    st.write(f"- Task: {extr['task_name']}")
                    st.write(f"  Status: {'Disabled' if extr['disabled'] else 'Enabled'}")
                    st.write(f"  Mode: {extr['mode']}")
                    if extr['filter']:
                        st.write("  Filter:")
                        st.code(extr['filter'], language='sql')
                    if extr['change_date_column']:
                        st.write(f"  Change Column: {extr['change_date_column']}")
                    st.markdown("---")

if __name__ == '__main__':
    main()