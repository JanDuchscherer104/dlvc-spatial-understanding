import argparse
import json
import subprocess
from pathlib import Path

ALLOWED_EXTENSIONS = {'.py', '.ipynb', '.md', '.tex', '.qmd'}
DIFF_LIMIT = 20000  # characters

def git(command, repo_path):
    result = subprocess.run(['git'] + command, cwd=repo_path,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout

def get_commit_hashes(repo_path, since=None):
    cmd = ['rev-list', '--reverse', 'HEAD']
    if since:
        cmd.insert(1, f'--since={since}')
    output = git(cmd, repo_path)
    return output.strip().splitlines()

def get_commit_info(commit_hash, repo_path):
    fmt = '%H%n%an <%ae>%n%cI%n%B'
    output = git(['show', '-s', f'--format={fmt}', commit_hash], repo_path)
    lines = output.splitlines()
    commit_hash = lines[0]
    author = lines[1]
    timestamp = lines[2]
    message = '\n'.join(lines[3:])
    return commit_hash, author, timestamp, message

def get_changed_files(commit_hash, repo_path):
    output = git(['diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash], repo_path)
    files = [f for f in output.splitlines() if Path(f).suffix in ALLOWED_EXTENSIONS]
    return files

def get_diff_for_file(commit_hash, file_path, repo_path):
    diff = git(['show', f'{commit_hash}', '--', file_path], repo_path)
    if len(diff) > DIFF_LIMIT:
        diff = diff[:DIFF_LIMIT] + '\n... [truncated]'
    return diff

def collect_commits(repo_path, since=None):
    commits = []
    for commit_hash in get_commit_hashes(repo_path, since):
        chash, author, timestamp, message = get_commit_info(commit_hash, repo_path)
        changed_files = get_changed_files(chash, repo_path)
        diffs_by_file = {f: get_diff_for_file(chash, f, repo_path) for f in changed_files}
        commits.append({
            'commit_hash': chash,
            'author': author,
            'timestamp': timestamp,
            'message': message,
            'changed_files': changed_files,
            'diffs_by_file': diffs_by_file,
        })
    return commits

def main():
    parser = argparse.ArgumentParser(description="Extract commit history to JSON")
    parser.add_argument('--output', default='commits.json', help='Output JSON file')
    parser.add_argument('--since', help='ISO date to filter commits')
    parser.add_argument('--repo-path', default='.', help='Path to git repository')
    args = parser.parse_args()

    commits = collect_commits(args.repo_path, args.since)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(commits, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
