/usr/bin/env python3
"""Web search integration using Strategy pattern."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request


class SearchEngine(ABC):
    """Abstract interface for search engines."""

    @abstractmethod
    def search(self, query: str) -> str:
        """Execute search and return formatted results."""
        pass


class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo implementation with response object pattern."""

    Url: str = 'https://html.duckduckgo.com/html/'
    MaxResults: int = 5

    def search(self, query: str) -> str:
        """Search DuckDuckGo and return results."""
        return self._fetch_results(query)

    def _fetch_results(self, query: str) -> str:
        """Fetch and parse HTML results from DuckDuckGo."""
        data = urllib.parse.urlencode({'q': query}).encode('utf-8')
        req = urllib.request.Request(
            self.Url,
            data=data,
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': 'https://html.duckduckgo.com/',
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                page = resp.read().decode('utf-8', errors='replace')
        except (urllib.error.URLError, OSError) as e:
            return f'Search error: {e}'

        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', page)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page)

        results: List[str] = []
        for i in range(min(len(titles), len(snippets), self.MaxResults)):
            title = html.unescape(re.sub(r'<[^>]+>', '', titles[i])).strip()
            snippet = html.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip()
            results.append(f"{i + 1}. {title}\n   {snippet}")

        if not results:
            return 'No search results found.'

        output = 'Search results:\n' + '\n'.join(results)
        return output


class WeatherWeatherEngine(SearchEngine):
    """Weather-specific search engine using Open-Meteo APIs."""

    GeoUrl: str = 'https://geocoding-api.open-meteo.com/v1/search'
    WxUrl: str = 'https://api.open-meteo.com/v1/forecast'

    def search(self, query: str) -> str:
        """Search for weather information using two-step geocoding + forecast."""
        return self._fetch_weather(query)

    def _fetch_weather(self, query: str) -> str:
        """Fetch weather data using Open-Meteo APIs."""
        return self._get_weather_from_geo_api(query)

    def _get_weather_from_geo_api(self, query: str) -> str:
        """Get weather from Open-Meteo geocoding API."""
        geo_url = self.GeoUrl + '?' + urllib.parse.urlencode({'name': query, 'count': 1})
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read().decode('utf-8'))
        results = geo.get('results', [])
        if not results:
            return ''
        lat, lon = results[0]['latitude'], results[0]['longitude']
        name = results[0].get('name', query)

        wx_url = self.WxUrl + '?' + urllib.parse.urlencode({
            'latitude': lat,
            'longitude': lon,
            'current_weather': 'true',
        })
        with urllib.request.urlopen(wx_url, timeout=10) as resp:
            wx = json.loads(resp.read().decode('utf-8'))

        cw = wx.get('current_weather', {})
        temp = cw.get('temperature')
        wind = cw.get('windspeed')
        desc = cw.get('weathercode')
        WMO = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
            55: 'Dense drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
            80: 'Slight rain showers', 81: 'Moderate rain showers',
            82: 'Violent rain showers', 95: 'Thunderstorm', 96: 'Thunderstorm with hail',
            99: 'Severe thunderstorm with hail',
        }
        condition = WMO.get(desc, f'Code {desc}') if desc is not None else 'Unknown'

        lines = [f'Current weather in {name}:']
        if temp is not None:
            lines.append(f'  Temperature: {temp}°C')
        lines.append(f'  Condition: {condition}')
        if wind is not None:
            lines.append(f'  Wind speed: {wind} km/h')
        return '\n'.join(lines)
