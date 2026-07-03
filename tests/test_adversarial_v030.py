"""Adversarial tests for v0.3.0 changes — registry, --model, rcconfig, api_format."""
import json, os, sys, tempfile, traceback

# Ensure we use the local development version, not the installed pip package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

errors = []

def check(desc, cond, detail=''):
    if not cond:
        errors.append(f'FAIL: {desc} {detail}')
        print(f'  [FAIL] {desc}')
    else:
        print(f'  [OK]   {desc}')


# ═══════════════════════════════════════════════════════════════════
# 1. providers.json 健壮性
# ═══════════════════════════════════════════════════════════════════
print('\n=== 1. providers.json robustness ===')

from optimizer.config import load_providers
providers = load_providers()
check('providers.json loaded', len(providers) > 0, f'got {len(providers)}')
check('deepseek present', any(p['name'] == 'deepseek' for p in providers))
check('anthropic present', any(p['name'] == 'anthropic' for p in providers))
check('openai present', any(p['name'] == 'openai' for p in providers))
check('groq present', any(p['name'] == 'groq' for p in providers))
check('each has required fields', all(
    'name' in p and 'api_format' in p and 'model_patterns' in p
    for p in providers
))
for p in providers:
    check(f'{p["name"]} api_format valid', p.get('api_format') in ('openai', 'anthropic'))


# ═══════════════════════════════════════════════════════════════════
# 2. resolve_model() 边界
# ═══════════════════════════════════════════════════════════════════
print('\n=== 2. resolve_model() edge cases ===')

from optimizer.config import resolve_model

info = resolve_model('deepseek-v4-flash')
check('deepseek-v4-flash resolves', info is not None)
if info:
    check('  provider=deepseek', info.provider == 'deepseek')
    check('  base_url not None', info.base_url is not None)
    check('  model_name preserved', info.model_name == 'deepseek-v4-flash')

# OpenAI family
check('gpt-4o -> openai', resolve_model('gpt-4o') and resolve_model('gpt-4o').provider == 'openai')
check('gpt-4-turbo -> openai', resolve_model('gpt-4-turbo') and resolve_model('gpt-4-turbo').provider == 'openai')
check('o1-preview -> openai', resolve_model('o1-preview') and resolve_model('o1-preview').provider == 'openai')

# Groq matches
check('llama-3.3 -> groq', resolve_model('llama-3.3-70b-versatile') and resolve_model('llama-3.3-70b-versatile').provider == 'groq')
check('mixtral-8x7b -> groq', resolve_model('mixtral-8x7b') and resolve_model('mixtral-8x7b').provider == 'groq')
check('deepseek-r1 -> groq', resolve_model('deepseek-r1-671b') and resolve_model('deepseek-r1-671b').provider == 'groq')

# xAI
check('grok-beta -> xai', resolve_model('grok-beta') and resolve_model('grok-beta').provider == 'xai')
check('grok-2 -> xai', resolve_model('grok-2-latest') and resolve_model('grok-2-latest').provider == 'xai')

# Unknown model
check('unknown model -> None', resolve_model('nonexistent-model-9000') is None)
check('empty string -> None', resolve_model('') is None)
try:
    resolve_model(None)
    check('None raises TypeError', False)
except (TypeError, AttributeError):
    check('None raises TypeError', True)


# ═══════════════════════════════════════════════════════════════════
# 3. infer_provider_from_url() 边缘
# ═══════════════════════════════════════════════════════════════════
print('\n=== 3. infer_provider_from_url() edge cases ===')

from optimizer.config import infer_provider_from_url

tests = [
    ('https://api.deepseek.com/anthropic', 'deepseek'),
    ('https://api.anthropic.com', 'anthropic'),
    ('https://api.openai.com/v1', 'openai'),
    ('https://my-resource.openai.azure.com', 'openai'),
    ('https://api.groq.com/openai/v1', 'groq'),
    ('https://api.together.xyz/v1', 'together'),
    ('https://api.mistral.ai/v1', 'mistral'),
    ('https://api.fireworks.ai/inference/v1', 'fireworks'),
    ('https://api.x.ai', 'xai'),
    ('https://api.perplexity.ai', 'perplexity'),
    ('https://api.githubcopilot.com', 'github-copilot'),
    ('https://openrouter.ai/api/v1', 'openrouter'),
    ('https://generativelanguage.googleapis.com', 'google-gemini'),
    ('https://unknown-custom.com/v1', None),
    ('', None),
]
for url, expected in tests:
    result = infer_provider_from_url(url)
    if expected is None:
        check(f'no match for {url}', result is None)
    else:
        check(f'{url} -> {expected}', result == expected, f'got {result}')


# ═══════════════════════════════════════════════════════════════════
# 4. get_config() backward compatibility
# ═══════════════════════════════════════════════════════════════════
print('\n=== 4. get_config() backward compat ===')

from optimizer.config import get_config, register_provider

for name in ['deepseek', 'anthropic', 'openai']:
    cfg = get_config(name)
    check(f'{name} get_config works', cfg.name == name)
    check(f'{name} has api_format', hasattr(cfg, 'api_format'))
    check(f'{name} cache_threshold > 0', cfg.cache_threshold > 0)

# New providers from registry
for name in ['groq', 'together', 'mistral', 'xai']:
    cfg = get_config(name)
    check(f'{name} from registry', cfg.name == name)
    check(f'{name} cache_threshold > 0', cfg.cache_threshold > 0)

# Runtime registration still works
register_provider('test-custom', cache_threshold=42, api_format='openai')
cfg = get_config('test-custom')
check('register_provider works', cfg.name == 'test-custom')
check('register_provider sets api_format', cfg.api_format == 'openai')
check('register_provider sets threshold', cfg.cache_threshold == 42)

# Unknown provider
cfg = get_config('totally-unknown')
check('unknown provider fallback', cfg.name == 'totally-unknown')
check('unknown provider default threshold', cfg.cache_threshold == 1024)

# api_format field exists on all configs
for name in list(get_config('deepseek').__dict__):
    pass
cm = get_config('deepseek')
check('api_format on deepseek', hasattr(cm, 'api_format'))
check('api_format on unknown', hasattr(get_config('nope'), 'api_format'))


# ═══════════════════════════════════════════════════════════════════
# 5. _resolve_upstream_provider() priority chain
# ═══════════════════════════════════════════════════════════════════
print('\n=== 5. _resolve_upstream_provider() priority chain ===')

from cli.main import _resolve_upstream_provider

# 5a: --model takes priority over env
os.environ['ANTHROPIC_BASE_URL'] = 'https://env-var.com'
u, p, f = _resolve_upstream_provider('', '', 'gpt-4o')
check('--model overrides env', 'openai.com' in u and p == 'openai')
del os.environ['ANTHROPIC_BASE_URL']

# 5b: --upstream overrides --model
u, p, f = _resolve_upstream_provider('https://manual.com', '', 'deepseek-v4-flash')
check('--upstream overrides --model', 'manual.com' in u)

# 5c: --provider overrides --model's provider
u, p, f = _resolve_upstream_provider('', 'anthropic', 'gpt-4o')
check('--provider overrides --model provider', p == 'anthropic')
check('  but model upstream still used', 'openai.com' in u)

# 5d: env upstream fallback
os.environ['POPT_UPSTREAM'] = 'https://popt-env.com/v1'
u, p, f = _resolve_upstream_provider('', '', '')
check('env POPT_UPSTREAM detected', 'popt-env.com' in u)
del os.environ['POPT_UPSTREAM']

# 5e: model via env
os.environ['POPT_MODEL'] = 'grok-beta'
u, p, f = _resolve_upstream_provider('', '', '')
check('POPT_MODEL env var resolves', 'x.ai' in u and p == 'xai')
del os.environ['POPT_MODEL']

# 5f: all empty -> empty strings with openai default
u, p, f = _resolve_upstream_provider('', '', '')
check('all empty returns empty strings', u == '' and p == '')
check('all empty default api_format', f == 'openai')


# ═══════════════════════════════════════════════════════════════════
# 6. rcconfig.py edge cases
# ═══════════════════════════════════════════════════════════════════
print('\n=== 6. rcconfig.py edge cases ===')

from cli.rcconfig import load_file, find_config_files, load_config

# No config files -> empty PopConfig
cfg = load_config()
check('no rc file -> empty cfg', cfg.model == '' and cfg.provider == '')

# TOML config
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8')
tmp.write('[project]\nmodel = "deepseek-v4-flash"\nprovider = "deepseek"\n\n[proxy]\nport = 8080\n')
tmp.close()
os.environ['POPT_CONFIG'] = tmp.name
cfg = load_config()
check('TOML model loaded', cfg.model == 'deepseek-v4-flash')
check('TOML provider loaded', cfg.provider == 'deepseek')
check('TOML port loaded', cfg.port == 8080)
check('TOML default host', cfg.host == '127.0.0.1')
del os.environ['POPT_CONFIG']
os.unlink(tmp.name)

# JSON config
tmp2 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
json.dump({'project': {'model': 'gpt-4o'}}, tmp2)
tmp2.close()
os.environ['POPT_CONFIG'] = tmp2.name
cfg = load_config()
check('JSON model loaded', cfg.model == 'gpt-4o')
del os.environ['POPT_CONFIG']
os.unlink(tmp2.name)

# Malformed file -> skip
tmp3 = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, encoding='utf-8')
tmp3.write('[[[ invalid toml')
tmp3.close()
os.environ['POPT_CONFIG'] = tmp3.name
cfg = load_config()
check('malformed file skipped gracefully', cfg.model == '')
del os.environ['POPT_CONFIG']
os.unlink(tmp3.name)

# No-extension file -> try TOML then JSON fallback
tmp4 = tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False, encoding='utf-8')
tmp4.write('{"project": {"model": "claude-3"}}')
tmp4.close()
os.environ['POPT_CONFIG'] = tmp4.name
cfg = load_config()
check('no-extension JSON fallback', cfg.model == 'claude-3')
del os.environ['POPT_CONFIG']
os.unlink(tmp4.name)

# Non-existent file -> no crash
os.environ['POPT_CONFIG'] = '/nonexistent/path/.poptimerc'
cfg = load_config()
check('non-existent file no crash', cfg.model == '')
del os.environ['POPT_CONFIG']

# get_config_value helper
from cli.rcconfig import get_config_value
# Without any config, should return default
val = get_config_value('model', 'default-model')
check('get_config_value returns default', val == 'default-model')


# ═══════════════════════════════════════════════════════════════════
# 7. formatter with api_format
# ═══════════════════════════════════════════════════════════════════
print('\n=== 7. formatter api_format ===')

from optimizer.formatter import Formatter
from optimizer.config import ProviderConfig

msgs = [{'role': 'system', 'content': 'hi'}, {'role': 'user', 'content': 'hello'}]

# anthropic api_format
result = Formatter(ProviderConfig(name='test', api_format='anthropic')).format(msgs)
check('anthropic format keeps system', any(m.get('role') == 'system' for m in result))

# openai api_format
result = Formatter(ProviderConfig(name='test', api_format='openai')).format(msgs)
check('openai format keeps user', any(m.get('role') == 'user' for m in result))

# unknown api_format -> just strip internal markers
msgs_m = [{'role': 'user', 'content': 'hello', '_role_type': 'user', '_is_separator': False}]
result = Formatter(ProviderConfig(name='test', api_format='unknown')).format(msgs_m)
check('unknown format strips _role_type', '_role_type' not in result[0])
check('unknown format strips _is_separator', '_is_separator' not in result[0])
check('unknown format keeps role', result[0]['role'] == 'user')

# empty api_format -> defaults to openai in format
result = Formatter(ProviderConfig(name='test', api_format='')).format(msgs)
check('empty api_format works', any(m.get('role') == 'user' for m in result))


# ═══════════════════════════════════════════════════════════════════
# 8. Optimize pipeline integration
# ═══════════════════════════════════════════════════════════════════
print('\n=== 8. optimize pipeline integration ===')

from optimizer import optimize

test_msgs = [
    {'role': 'system', 'content': 'Be concise.'},
    {'role': 'user', 'content': 'Hello'},
]

for provider in ['deepseek', 'anthropic', 'openai', 'groq']:
    result = optimize(test_msgs, provider=provider)
    check(f'{provider} optimize works', len(result) >= 2)
    for m in result:
        check(f'{provider} no _role_type leak', '_role_type' not in m)
        check(f'{provider} no _is_separator leak', '_is_separator' not in m)

# Idempotency
r1 = optimize(test_msgs, provider='openai')
r2 = optimize(r1, provider='openai')
r1_clean = [(m.get('role',''), m.get('content','')) for m in r1]
r2_clean = [(m.get('role',''), m.get('content','')) for m in r2]
check('optimize idempotent', r1_clean == r2_clean)


# ═══════════════════════════════════════════════════════════════════
# 9. CLI argument parsing
# ═══════════════════════════════════════════════════════════════════
print('\n=== 9. CLI argument parsing ===')

from cli.main import build_parser

p = build_parser()

# --model flag parsing
for cmd in ['run', 'proxy']:
    args = p.parse_args([cmd, '--model', 'deepseek-v4-flash'])
    check(f'{cmd} --model deepseek', args.model == 'deepseek-v4-flash')

    args = p.parse_args([cmd, '-m', 'gpt-4o'])
    check(f'{cmd} -m gpt-4o', args.model == 'gpt-4o')

# --model with --upstream
args = p.parse_args(['run', '--model', 'gpt-4o', '--upstream', 'https://custom.com', '--', 'echo', 'hi'])
check('--model + --upstream parsed', args.model == 'gpt-4o' and args.upstream == 'https://custom.com')

# --model with --provider
args = p.parse_args(['proxy', '--model', 'deepseek-v4-flash', '--provider', 'anthropic'])
check('--model + --provider parsed', args.model == 'deepseek-v4-flash' and args.provider == 'anthropic')

# Empty --model
args = p.parse_args(['run', 'echo', 'hello'])
check('no --model, no problem', hasattr(args, 'model'))


# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
print()
print('========================================')
if errors:
    print(f'FAILED: {len(errors)} tests:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('ALL ADVERSARIAL TESTS PASSED')
    print('========================================')
