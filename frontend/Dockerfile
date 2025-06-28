FROM docker-reg.oss.net.bd/land_php73:v1
#apm
RUN apt update && apt install -y wget curl && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/elastic/apm-agent-php/releases/download/v1.15.0/apm-agent-php_1.15.0_amd64.deb && \
    dpkg -i apm-agent-php_1.15.0_amd64.deb

WORKDIR /var/www/html/
#RUN rm -rf ./*
COPY . .

RUN chown -R www-data:www-data /var/www/html
#RUN php -r "readfile('http://getcomposer.org/installer');" | php -- --install-dir=/usr/bin/ --filename=composer
COPY default.conf /etc/apache2/sites-enabled/000-default.conf

CMD ["/usr/sbin/apache2ctl", "-D", "FOREGROUND"]
EXPOSE 80 443