import os
import json

def merge_itm_results(folder):
    """
    For each pair of JSON files in the folder, where one is regular and one is ITM_only,
    merge the ITM_only results into the regular file by appending ITM results after the regular
    results for each (dataset, template_count) pair.
    """
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    # Map base name (without ITM_only) to files
    base_map = {}
    for f in files:
        if f.endswith('ITM_only.json'):
            base = f.replace('ITM_only.json', '')
            base_map.setdefault(base, {})['itm'] = f
        elif f.endswith('.json'):
            base = f.replace('.json', '')
            # Avoid double-matching ITM_only
            if not base.endswith('-ITM_only'):
                base_map.setdefault(base, {})['regular'] = f

    for base, pair in base_map.items():
        if 'regular' in pair and 'itm' in pair:
            regular_path = os.path.join(folder, pair['regular'])
            itm_path = os.path.join(folder, pair['itm'])

            with open(regular_path, 'r') as f:
                regular_json = json.load(f)
            with open(itm_path, 'r') as f:
                itm_json = json.load(f)

            # Build a lookup for ITM results: (dataset, template_count) -> itm_result
            itm_lookup = {}
            for itm_res in itm_json['results']:
                key = (itm_res['dataset'], itm_res['template_count'])
                itm_lookup[key] = itm_res

            # Build a new results list: for each regular result, append the ITM result after it (if exists)
            new_results = []
            for reg_res in regular_json['results']:
                key = (reg_res['dataset'], reg_res['template_count'])
                new_results.append(reg_res)
                if key in itm_lookup:
                    new_results.append(itm_lookup[key])

            # Optionally, add any ITM results that did not have a regular counterpart
            # for key, itm_res in itm_lookup.items():
            #     if not any((r['dataset'], r['template_count']) == key for r in regular_json['results']):
            #         new_results.append(itm_res)

            # Overwrite the regular file with merged results
            regular_json['results'] = new_results
            with open(regular_path, 'w') as f:
                json.dump(regular_json, f, indent=2)
            print(f"Merged ITM results into {pair['regular']}")

if __name__ == "__main__":
    # Example usage: merge_itm_results("/home/phisch/multimodal/test_results/full_dataset_aug_mlm/template_sets_and_itm")
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    merge_itm_results(folder)