#PS1="%D{%H:%M} %{${fg[red]}%}%n%{${reset_color}%}@%{${fg[yellow]}%}%~%{${reset_color}%}%% "
#PS1="%D{%H:%M} %/ %% "
#PS1="%F{red}%D{%H:%M}%f %~ %% " #可以省略一些空间
#PS1="%F{red}%D{%H:%M}%f %F{green}%~%f %% " #只显示时间没有具体日期
#PS1="%F{red}%D{%Y-%m-%d %H:%M}%f %F{green}%~%f %% "

export PS1="\[\033[0;36m\]\A \[\033[0;32m\]\w\[\033[0m\] \$ "


source setup_env.sh

