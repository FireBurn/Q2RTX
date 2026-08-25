#!/bin/bash

# The overrides are useful for portable installs and make this otherwise
# package-oriented launcher testable without writing under /usr.
BIN_PREFIX="${Q2RTX_BIN_DIR:-.}"
DATA_PREFIX="${Q2RTX_DATA_DIR:-}"

mkdir -p "${XDG_DATA_HOME:="${HOME}/.local/share"}"

# If the game is installed via a package manager q2rtx won't be in the same
# directory as q2rtx.sh
if [[ -z "${DATA_PREFIX}" && -d "/usr/share/quake2rtx" ]]; then
	DATA_PREFIX="/usr/share/quake2rtx"
fi
if [[ "${BIN_PREFIX}" == "." && "${DATA_PREFIX}" == "/usr/share/quake2rtx" ]]; then
	BIN_PREFIX="${DATA_PREFIX}/bin"
fi

if [[ -d "${HOME}/.quake2rtx"  && ! -d "${XDG_DATA_HOME}/quake2rtx" ]]; then
        mv "${HOME}/.quake2rtx" "${XDG_DATA_HOME}/quake2rtx"
fi


# Generate the user's game dir if doesn't exist
if [[ ! -d "${XDG_DATA_HOME}/quake2rtx/baseq2" ]]; then
	mkdir -p "${XDG_DATA_HOME}/quake2rtx/baseq2"
fi

# A user-local q2rtx_media.pkz is searched before the packaged archive.  Keep
# this small, source-controlled menu as a loose user override so upgrades can
# add new renderer cvar values (notably FSR3.1.5) without requiring users to
# delete their local media archive.  Preserve an older custom menu once; set
# Q2RTX_SKIP_MENU_UPDATE=1 to keep managing a custom menu manually.
MENU_REVISION="Q2RTX menu revision: 2026-08-fsr3-fsr4-v07-rr"
MENU_SOURCE="${DATA_PREFIX:+${DATA_PREFIX}/baseq2/q2rtx.menu}"
MENU_DEST="${XDG_DATA_HOME}/quake2rtx/baseq2/q2rtx.menu"
if [[ "${Q2RTX_SKIP_MENU_UPDATE:-0}" != "1" && -n "${MENU_SOURCE}" && -f "${MENU_SOURCE}" ]]; then
	if [[ ! -f "${MENU_DEST}" ]] || ! grep -Fq "${MENU_REVISION}" "${MENU_DEST}"; then
		if [[ -f "${MENU_DEST}" && ! -f "${MENU_DEST}.pre-fsr-menu-update" ]]; then
			cp -p "${MENU_DEST}" "${MENU_DEST}.pre-fsr-menu-update"
		fi
		cp -p "${MENU_SOURCE}" "${MENU_DEST}"
	fi
fi

# Only run this script on first-launch
if [[ ! -f "${XDG_DATA_HOME}/quake2rtx/.retail_checked" ]]; then
	${BIN_PREFIX}/find-retail-paks.sh
	touch ${XDG_DATA_HOME}/quake2rtx/.retail_checked
fi

${BIN_PREFIX}/q2rtx "$@"
