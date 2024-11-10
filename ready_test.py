import sys
import json
import logging
import random
import time
import uuid
import platform
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from urllib.parse import urljoin
from fake_useragent import UserAgent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QFileDialog,
    QSpinBox, QGroupBox, QLineEdit, QRadioButton, QButtonGroup,
    QComboBox, QGridLayout  # Добавлен QGridLayout здесь
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class DeviceProfile:
    """Расширенный профиль устройства с дополнительными параметрами"""
    
    def __init__(self):
        self.gpu_vendors = [
            {'vendor': 'NVIDIA', 'models': ['GTX 1660', 'RTX 2060', 'RTX 3070', 'RTX 3080']},
            {'vendor': 'AMD', 'models': ['RX 580', 'RX 5700', 'RX 6800']},
            {'vendor': 'Intel', 'models': ['UHD 620', 'Iris Xe', 'HD 4000']}
        ]
        
        self.connection_types = ['wifi', '4g', '3g', 'ethernet']
        self.battery_levels = list(range(20, 101))
        self.common_fonts = [
            'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
            'Verdana', 'Georgia', 'Tahoma', 'Trebuchet MS'
        ]
        
    def generate_device_info(self) -> Dict[str, Any]:
        """Генерирует информацию об устройстве"""
        gpu_info = random.choice(self.gpu_vendors)
        return {
            'gpu': {
                'vendor': gpu_info['vendor'],
                'model': random.choice(gpu_info['models']),
                'webgl_params': self._generate_webgl_params()
            },
            'battery': {
                'level': random.choice(self.battery_levels),
                'charging': random.choice([True, False])
            },
            'connection': {
                'type': random.choice(self.connection_types),
                'downlink': random.uniform(1.0, 15.0),
                'rtt': random.randint(50, 200)
            },
            'fonts': random.sample(self.common_fonts, random.randint(5, 10))
        }
        
    def _generate_webgl_params(self) -> Dict[str, Any]:
        """Генерирует параметры WebGL"""
        return {
            'max_texture_size': random.choice([4096, 8192, 16384]),
            'max_viewport_dims': random.choice([(4096, 4096), (8192, 8192), (16384, 16384)]),
            'extensions': self._generate_webgl_extensions()
        }
        
    def _generate_webgl_extensions(self) -> List[str]:
        """Генерирует список поддерживаемых расширений WebGL"""
        base_extensions = [
            'ANGLE_instanced_arrays',
            'EXT_blend_minmax',
            'EXT_color_buffer_half_float',
            'EXT_disjoint_timer_query',
            'EXT_float_blend',
            'EXT_frag_depth',
            'EXT_shader_texture_lod'
        ]
        return random.sample(base_extensions, random.randint(4, len(base_extensions)))

class BrowserProfile:
    def __init__(self):
        self.ua = UserAgent()
        self.languages = ['en-US', 'en-GB', 'ru-RU', 'de-DE', 'fr-FR', 'es-ES']
        self.platforms = ['Windows', 'MacOS', 'Linux']
        self.screen_resolutions = [
            '1920x1080', '1366x768', '1536x864', '1440x900',
            '1280x720', '2560x1440', '3840x2160'
        ]
        self.color_depths = [24, 32]
        self.timezone_offsets = list(range(-12, 13))
        self.webgl_vendors = [
            'Google Inc.', 'Intel Inc.', 'NVIDIA Corporation',
            'AMD', 'Apple Inc.'
        ]
        self.device_profile = DeviceProfile()
        
    def generate_profile(self) -> Dict:
        """Генерирует расширенный профиль браузера"""
        device_info = self.device_profile.generate_device_info()
        
        profile = {
            'id': str(uuid.uuid4()),
            'userAgent': self.ua.random,
            'acceptLanguage': random.choice(self.languages),
            'platform': random.choice(self.platforms),
            'screenResolution': random.choice(self.screen_resolutions),
            'colorDepth': random.choice(self.color_depths),
            'timezoneOffset': random.choice(self.timezone_offsets),
            'webglVendor': random.choice(self.webgl_vendors),
            'device': device_info,
            'fingerprints': self._generate_fingerprints(),
            'storage': self._generate_storage_data(),
            'network': self._generate_network_profile()
        }
        return profile
        
    def _generate_fingerprints(self) -> Dict[str, str]:
        """Генерирует уникальные отпечатки браузера"""
        return {
            'canvas': hashlib.md5(str(random.random()).encode()).hexdigest(),
            'webgl': hashlib.md5(str(random.random()).encode()).hexdigest(),
            'audio': hashlib.md5(str(random.random()).encode()).hexdigest(),
            'clientRects': hashlib.md5(str(random.random()).encode()).hexdigest(),
            'fonts': hashlib.md5(str(random.random()).encode()).hexdigest()
        }
        
    def _generate_storage_data(self) -> Dict[str, Dict]:
        """Генерирует данные для локального хранилища и cookie"""
        return {
            'localStorage': self._generate_local_storage(),
            'sessionStorage': self._generate_session_storage(),
            'cookies': self._generate_cookies()
        }
        
    def _generate_local_storage(self) -> Dict[str, str]:
        """Генерирует данные Local Storage"""
        items = {
            'theme': random.choice(['light', 'dark', 'system']),
            'language': random.choice(self.languages),
            'lastVisit': str(int(time.time()) - random.randint(0, 86400)),
            'preferences': json.dumps({
                'notifications': random.choice([True, False]),
                'autoplay': random.choice([True, False])
            })
        }
        return items
        
    def _generate_session_storage(self) -> Dict[str, str]:
        """Генерирует данные Session Storage"""
        return {
            'sessionId': str(uuid.uuid4()),
            'tabId': str(random.randint(1, 100)),
            'lastAction': str(int(time.time()))
        }
        
    def _generate_cookies(self) -> List[Dict]:
        """Генерирует cookie"""
        return [
            {
                'name': 'session_id',
                'value': str(uuid.uuid4()),
                'domain': '.yandex.ru',
                'path': '/',
                'expires': int(time.time()) + 86400
            },
            {
                'name': 'user_preferences',
                'value': hashlib.md5(str(random.random()).encode()).hexdigest(),
                'domain': '.yandex.ru',
                'path': '/',
                'expires': int(time.time()) + 86400 * 30
            }
        ]
        
    def _generate_network_profile(self) -> Dict[str, Any]:
        """Генерирует профиль сетевого поведения"""
        return {
            'requestDelay': {
                'min': random.uniform(0.1, 0.5),
                'max': random.uniform(1.0, 2.0)
            },
            'loadingPriorities': self._generate_loading_priorities(),
            'cacheSettings': self._generate_cache_settings()
        }
        
    def _generate_loading_priorities(self) -> Dict[str, int]:
        """Генерирует приоритеты загрузки ресурсов"""
        return {
            'html': 1,
            'css': random.randint(2, 3),
            'javascript': random.randint(2, 4),
            'images': random.randint(3, 5),
            'fonts': random.randint(4, 6)
        }
        
    def _generate_cache_settings(self) -> Dict[str, Any]:
        """Генерирует настройки кэширования"""
        return {
            'maxAge': random.randint(3600, 86400),
            'revalidate': random.choice([True, False]),
            'strategies': ['memory', 'disk']
        }

class ProxyManager:
    def __init__(self, proxy_file: Optional[str] = None):
        self.proxy_file = proxy_file
        self.proxies = self._load_proxies() if proxy_file else []
        self.working_proxies = set()
        self.failed_proxies = set()
        self.proxy_stats = {}
        
    def _load_proxies(self) -> List[Dict]:
        """Загружает список прокси из файла"""
        try:
            with open(self.proxy_file, 'r') as f:
                proxies = []
                for line in f:
                    try:
                        proxy_str = line.strip()
                        if ':' not in proxy_str:
                            continue
                        
                        parts = proxy_str.split(':')
                        if len(parts) == 2:
                            host, port = parts
                            proxy = {
                                'http': f'http://{host}:{port}',
                                'https': f'http://{host}:{port}'
                            }
                        elif len(parts) == 4:
                            host, port, username, password = parts
                            proxy = {
                                'http': f'http://{username}:{password}@{host}:{port}',
                                'https': f'http://{username}:{password}@{host}:{port}'
                            }
                        proxies.append(proxy)
                    except Exception as e:
                        logging.error(f"Ошибка при разборе строки прокси: {str(e)}")
                        continue
                return proxies
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла прокси: {str(e)}")
            return []
            
    def check_proxy(self, proxy: Dict) -> bool:
        """Проверяет работоспособность прокси"""
        try:
            response = requests.get(
                'https://yandex.ru',
                proxies=proxy,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
            
    def get_working_proxy(self) -> Optional[Dict]:
        """Возвращает работающий прокси"""
        if not self.proxies:
            return None
            
        # Пробуем использовать уже проверенные прокси
        working_proxies = list(self.working_proxies - self.failed_proxies)
        if working_proxies:
            return random.choice(working_proxies)
            
        # Проверяем новые прокси
        for proxy in self.proxies:
            if proxy not in self.failed_proxies and self.check_proxy(proxy):
                self.working_proxies.add(proxy)
                return proxy
                
        return None
        
    def mark_proxy_failed(self, proxy: Dict):
        """Помечает прокси как неработающий"""
        if proxy in self.working_proxies:
            self.working_proxies.remove(proxy)
        self.failed_proxies.add(proxy)
        
    def update_proxy_stats(self, proxy: Dict, success: bool):
        """Обновляет статистику использования прокси"""
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {'success': 0, 'failed': 0}
        
        if success:
            self.proxy_stats[proxy]['success'] += 1
        else:
            self.proxy_stats[proxy]['failed'] += 1

class HumanEmulator:
    """Эмулятор человеческого поведения"""
    
    @staticmethod
    def get_typing_delay() -> float:
        """Возвращает задержку между нажатиями клавиш"""
        return random.uniform(0.1, 0.3)
        
    @staticmethod
    def get_mouse_movement() -> List[Dict[str, int]]:
        """Генерирует траекторию движения мыши"""
        points = []
        x, y = 0, 0
        for _ in range(random.randint(5, 15)):
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            points.append({'x': x, 'y': y})
        return points
        
    @staticmethod
    def get_scroll_pattern() -> List[Dict[str, int]]:
        """Генерирует паттерн скроллинга"""
        patterns = []
        position = 0
        for _ in range(random.randint(3, 8)):
            speed = random.randint(100, 300)
            distance = random.randint(300, 800)
            position += distance
            patterns.append({
                'position': position,
                'speed': speed,
                'pause': random.uniform(0.5, 2.0)
            })
        return patterns

class BehaviorManager:
    """Управление поведенческими паттернами"""
    
    def __init__(self):
        self.human_emulator = HumanEmulator()
        
    def apply_human_behavior(self, session: requests.Session):
        """Применяет человеческое поведение к сессии"""
        # Эмуляция задержек между действиями
        time.sleep(random.uniform(1.0, 3.0))
        
        # Эмуляция движений мыши
        mouse_movements = self.human_emulator.get_mouse_movement()
        
        # Эмуляция скроллинга
        scroll_patterns = self.human_emulator.get_scroll_pattern()
        
        # Эмуляция задержек при вводе
        typing_delays = [self.human_emulator.get_typing_delay() 
                        for _ in range(random.randint(5, 15))]
        
        return {
            'mouse_movements': mouse_movements,
            'scroll_patterns': scroll_patterns,
            'typing_delays': typing_delays
        }

class CaptchaHandler:
    """Обработчик капчи"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.solved_captchas = {}
        
    def solve_captcha(self, captcha_type: str, captcha_data: str) -> Optional[str]:
        """Решает капчу используя внешний сервис"""
        # Проверяем кэш решенных капч
        cache_key = hashlib.md5(captcha_data.encode()).hexdigest()
        if cache_key in self.solved_captchas:
            return self.solved_captchas[cache_key]
            
        # Здесь должна быть интеграция с сервисом решения капчи
        # Примерная реализация:
        try:
            # solution = anti_captcha_service.solve(captcha_data, self.api_key)
            # self.solved_captchas[cache_key] = solution
            # return solution
            pass
        except Exception as e:
            logging.error(f"Ошибка при решении капчи: {str(e)}")
            return None

class MessengerWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, messenger, urls: List[str], delay_min: int, delay_max: int, max_threads: int):
        super().__init__()
        self.messenger = messenger
        self.urls = urls
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_threads = max_threads
        self.is_running = True
        self.browser_profile_generator = BrowserProfile()
        self.behavior_manager = BehaviorManager()
        self.proxy_manager = ProxyManager()
        self.captcha_handler = CaptchaHandler()
        self.thread_profiles = {}
        self.thread_stats = {}
        
    def get_thread_profile(self, thread_id: int) -> Dict:
        """Получает или создает профиль для потока"""
        if thread_id not in self.thread_profiles:
            profile = self.browser_profile_generator.generate_profile()
            self.thread_profiles[thread_id] = profile
            self.thread_stats[thread_id] = {
                'requests': 0,
                'success': 0,
                'failed': 0,
                'captchas': 0,
                'proxy_changes': 0
            }
        return self.thread_profiles[thread_id]

    def process_url(self, url: str) -> bool:
        try:
            thread_id = int(QThread.currentThreadId())
            profile = self.get_thread_profile(thread_id)
            
            self.log.emit(f"Обработка: {url} (Thread ID: {thread_id})")
            self.log.emit(f"Используется профиль: {profile['id']}")
            
            # Применяем поведенческие паттерны
            behavior = self.behavior_manager.apply_human_behavior(requests.Session())
            
            # Получаем рабочий прокси
            proxy = self.proxy_manager.get_working_proxy()
            if not proxy:
                self.log.emit("Не удалось найти рабочий прокси")
                return False
                
            # Пытаемся отправить сообщение
            max_attempts = 3
            for attempt in range(max_attempts):
                result = self.messenger.send_message(
                    url, 
                    profile,
                    proxy=proxy,
                    behavior=behavior
                )
                
                if result.get('success'):
                    self.update_stats(thread_id, True)
                    self.proxy_manager.update_proxy_stats(proxy, True)
                    self.log.emit("✓ Успешно отправлено")
                    return True
                    
                elif result.get('captcha_required'):
                    self.log.emit("Обнаружена капча, пытаемся решить...")
                    captcha_solution = self.captcha_handler.solve_captcha(
                        result['captcha_type'],
                        result['captcha_data']
                    )
                    if captcha_solution:
                        result = self.messenger.send_message(
                            url,
                            profile,
                            proxy=proxy,
                            captcha_solution=captcha_solution
                        )
                        if result.get('success'):
                            self.update_stats(thread_id, True)
                            self.log.emit("✓ Сообщение отправлено после решения капчи")
                            return True
                            
                elif result.get('proxy_blocked'):
                    self.log.emit("Прокси заблокирован, меняем...")
                    self.proxy_manager.mark_proxy_failed(proxy)
                    proxy = self.proxy_manager.get_working_proxy()
                    if not proxy:
                        self.log.emit("Не удалось найти новый рабочий прокси")
                        return False
                        
                self.update_stats(thread_id, False)
                
            self.log.emit("✗ Превышено количество попыток")
            return False
            
        except Exception as e:
            self.log.emit(f"Критическая ошибка: {str(e)}")
            return False

    def update_stats(self, thread_id: int, success: bool):
        """Обновляет статистику потока"""
        stats = self.thread_stats[thread_id]
        stats['requests'] += 1
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

    def run(self):
        total_urls = len(self.urls)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_url = {
                executor.submit(self.process_url, url): url 
                for url in self.urls
            }

            for future in as_completed(future_to_url):
                if not self.is_running:
                    for f in future_to_url:
                        f.cancel()
                    break

                completed += 1
                self.status.emit(f"Обработка {completed}/{total_urls}")
                self.progress.emit(int((completed / total_urls) * 100))

                if completed < total_urls and self.is_running:
                    delay = random.uniform(self.delay_min, self.delay_max)
                    self.log.emit(f"Ожидание {delay:.1f} сек...")
                    time.sleep(delay)

        # Собираем итоговую статистику
        results = {
            "total": total_urls,
            "success": sum(stats['success'] for stats in self.thread_stats.values()),
            "failed": sum(stats['failed'] for stats in self.thread_stats.values()),
            "captchas_solved": sum(stats['captchas'] for stats in self.thread_stats.values()),
            "proxy_changes": sum(stats['proxy_changes'] for stats in self.thread_stats.values())
        }
        self.finished.emit(results)

    def stop(self):
        self.is_running = False

class YandexServicesMessenger:
    def __init__(self, proxy_file: Optional[str] = None, messages_file: Optional[str] = None):
        self.proxy_file = proxy_file
        self.messages_file = messages_file
        self.proxies = self._load_proxies() if proxy_file else None
        self.messages = self._load_messages() if messages_file else None
        self.current_proxy_index = 0
        self.current_message_index = 0
        self.behavior_manager = BehaviorManager()
        self.proxy_manager = ProxyManager(proxy_file)
        self.captcha_handler = CaptchaHandler()
        
    def _load_proxies(self) -> List[dict]:
        """Загружает список прокси из файла"""
        return self.proxy_manager._load_proxies()

    def _load_messages(self) -> List[str]:
        """Загружает сообщения из файла"""
        try:
            if self.messages_file.endswith('.txt'):
                with open(self.messages_file, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            elif self.messages_file.endswith('.xlsx'):
                df = pd.read_excel(self.messages_file)
                return df.iloc[:, 0].tolist()
            return []
        except Exception as e:
            logging.error(f"Ошибка при загрузке сообщений: {str(e)}")
            return []

    def _get_next_message(self) -> Optional[str]:
        """Получает следующее сообщение из списка"""
        if not self.messages:
            return None
        message = self.messages[self.current_message_index]
        self.current_message_index = (self.current_message_index + 1) % len(self.messages)
        return message

    def send_message(self, url: str, profile: Dict, proxy: Optional[Dict] = None,
                    behavior: Optional[Dict] = None, captcha_solution: Optional[str] = None) -> Dict:
        """
        Отправка сообщения на Яндекс.Услуги с расширенными параметрами
        
        Args:
            url: URL объявления
            profile: Профиль браузера
            proxy: Прокси-сервер
            behavior: Поведенческие паттерны
            captcha_solution: Решение капчи
            
        Returns:
            Dict: Результат отправки с дополнительной информацией
        """
        try:
            message = self._get_next_message()
            if not message:
                return {'success': False, 'error': 'Нет доступных сообщений'}

            try:
                ad_id = url.split('/')[-1].split('-')[0]
            except:
                return {'success': False, 'error': f'Невозможно извлечь ID из URL: {url}'}

            api_url = f"https://uslugi.yandex.ru/api/offers/{ad_id}/chat"
            
            payload = {
                "message": message,
                "offerId": ad_id
            }
            
            if captcha_solution:
                payload['captcha_solution'] = captcha_solution

            session = requests.Session()
            
            if proxy:
                session.proxies.update(proxy)
            
            # Применяем профиль браузера
            headers = self._generate_headers(profile)
            session.headers.update(headers)
            
            # Применяем поведенческие паттерны
            if behavior:
                for movement in behavior.get('mouse_movements', []):
                    time.sleep(random.uniform(0.1, 0.3))
                
                for scroll in behavior.get('scroll_patterns', []):
                    time.sleep(scroll['pause'])
                
                for delay in behavior.get('typing_delays', []):
                    time.sleep(delay)
            
            # Эмулируем предварительную загрузку ресурсов
            self._preload_resources(session, url)
            
            response = session.post(
                api_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return {'success': True}
            
            elif response.status_code == 403:
                if 'captcha' in response.text.lower():
                    return {
                        'success': False,
                        'captcha_required': True,
                        'captcha_type': self._detect_captcha_type(response.text),
                        'captcha_data': response.text
                    }
                else:
                    return {'success': False, 'proxy_blocked': True}
            
            else:
                return {
                    'success': False,
                    'error': f'Ошибка {response.status_code}: {response.text}'
                }

        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Ошибка сети: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Непредвиденная ошибка: {str(e)}'}

    def _generate_headers(self, profile: Dict) -> Dict:
        """Генерирует заголовки запроса на основе профиля браузера"""
        headers = {
            'User-Agent': profile['userAgent'],
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': profile['acceptLanguage'],
            'Content-Type': 'application/json;charset=UTF-8',
            'Origin': 'https://yandex.ru',
            'Referer': 'https://yandex.ru/',
            'Sec-Ch-Ua-Platform': f'"{profile["platform"]}"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Client-Data': profile['fingerprints']['canvas'],
            'X-WebGL-Data': profile['fingerprints']['webgl'],
            'X-Audio-Data': profile['fingerprints']['audio'],
            'X-Battery-Data': json.dumps(profile['device']['battery']),
            'X-Connection-Data': json.dumps(profile['device']['connection']),
            'X-Timezone-Offset': str(profile['timezoneOffset']),
        }
        return headers

    def _preload_resources(self, session: requests.Session, url: str):
        """Эмулирует предварительную загрузку ресурсов"""
        try:
            # Загружаем главную страницу
            session.get(url, timeout=10)
            
            # Эмулируем загрузку общих ресурсов
            common_resources = [
                '/api/offers/view',
                '/api/users/profile',
                '/api/chat/status'
            ]
            
            for resource in common_resources:
                try:
                    session.get(
                        urljoin('https://uslugi.yandex.ru', resource),
                        timeout=5
                    )
                    time.sleep(random.uniform(0.1, 0.5))
                except:
                    continue
                    
        except:
            pass

    def _detect_captcha_type(self, response_text: str) -> str:
        """Определяет тип капчи из ответа сервера"""
        if 'recaptcha' in response_text.lower():
            return 'recaptcha'
        elif 'smartcaptcha' in response_text.lower():
            return 'smartcaptcha'
        return 'unknown'

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    handlers = logging.StreamHandler(sys.stdout)
    handlers.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger = logging.getLogger()
    logger.addHandler(handlers)
    logger.setLevel(logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yandex Services Messenger")
        self.setMinimumSize(800, 600)
        
        # Основной виджет и layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Группа настроек файлов
        files_group = QGroupBox("Файлы")
        files_layout = QGridLayout()
        
        self.urls_path = QLineEdit()
        self.urls_button = QPushButton("Выбрать URLs")
        self.urls_button.clicked.connect(lambda: self._select_file(self.urls_path, "URLs (*.txt *.xlsx)"))
        
        self.messages_path = QLineEdit()
        self.messages_button = QPushButton("Выбрать сообщения")
        self.messages_button.clicked.connect(lambda: self._select_file(self.messages_path, "Messages (*.txt *.xlsx)"))
        
        self.proxies_path = QLineEdit()
        self.proxies_button = QPushButton("Выбрать прокси")
        self.proxies_button.clicked.connect(lambda: self._select_file(self.proxies_path, "Proxies (*.txt)"))
        
        files_layout.addWidget(QLabel("URLs:"), 0, 0)
        files_layout.addWidget(self.urls_path, 0, 1)
        files_layout.addWidget(self.urls_button, 0, 2)
        
        files_layout.addWidget(QLabel("Сообщения:"), 1, 0)
        files_layout.addWidget(self.messages_path, 1, 1)
        files_layout.addWidget(self.messages_button, 1, 2)
        
        files_layout.addWidget(QLabel("Прокси:"), 2, 0)
        files_layout.addWidget(self.proxies_path, 2, 1)
        files_layout.addWidget(self.proxies_button, 2, 2)
        
        files_group.setLayout(files_layout)
        
        # Группа настроек работы
        settings_group = QGroupBox("Настройки")
        settings_layout = QGridLayout()
        
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 50)
        self.threads_spin.setValue(5)
        
        self.delay_min_spin = QSpinBox()
        self.delay_min_spin.setRange(0, 3600)
        self.delay_min_spin.setValue(30)
        
        self.delay_max_spin = QSpinBox()
        self.delay_max_spin.setRange(0, 3600)
        self.delay_max_spin.setValue(60)
        
        settings_layout.addWidget(QLabel("Потоков:"), 0, 0)
        settings_layout.addWidget(self.threads_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("Мин. задержка (сек):"), 1, 0)
        settings_layout.addWidget(self.delay_min_spin, 1, 1)
        
        settings_layout.addWidget(QLabel("Макс. задержка (сек):"), 2, 0)
        settings_layout.addWidget(self.delay_max_spin, 2, 1)
        
        settings_group.setLayout(settings_layout)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Запустить")
        self.start_button.clicked.connect(self.start_processing)
        
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готов к работе")
        
        # Лог
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Добавляем все элементы на главный layout
        layout.addWidget(files_group)
        layout.addWidget(settings_group)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_text)
        
        self.worker = None

    def _select_file(self, line_edit: QLineEdit, file_filter: str):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)

    def _load_urls(self) -> List[str]:
        """Загружает URLs из файла"""
        try:
            file_path = self.urls_path.text()
            if not file_path:
                raise ValueError("Не выбран файл с URLs")
                
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                return df.iloc[:, 0].tolist()
            else:
                raise ValueError("Неподдерживаемый формат файла")
                
        except Exception as e:
            self.log_text.append(f"Ошибка при загрузке URLs: {str(e)}")
            return []

    def start_processing(self):
        urls = self._load_urls()
        if not urls:
            self.log_text.append("Нет URLs для обработки")
            return
            
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        messenger = YandexServicesMessenger(
            proxy_file=self.proxies_path.text(),
            messages_file=self.messages_path.text()
        )
        
        self.worker = MessengerWorker(
            messenger,
            urls,
            self.delay_min_spin.value(),
            self.delay_max_spin.value(),
            self.threads_spin.value()
        )
        
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.log.connect(self.log_text.append)
        self.worker.finished.connect(self._on_finished)
        
        self.worker.start()

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Останавливаем...")

    def _on_finished(self, results: dict):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        success_rate = (results['success'] / results['total']) * 100 if results['total'] > 0 else 0
        
        self.log_text.append("\nИтоговая статистика:")
        self.log_text.append(f"Всего обработано: {results['total']}")
        self.log_text.append(f"Успешно: {results['success']}")
        self.log_text.append(f"Ошибок: {results['failed']}")
        self.log_text.append(f"Процент успеха: {success_rate:.1f}%")
        self.log_text.append(f"Решено капч: {results['captchas_solved']}")
        self.log_text.append(f"Смен прокси: {results['proxy_changes']}")
        
        self.status_label.setText("Готов к работе")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())